import os
import logging
import argparse
import joblib
import optuna
import pandas as pd
import numpy as np

from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder

os.makedirs("./data", exist_ok=True)
logging.basicConfig(
    filename="./data/log_file.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class My_Classifier_Model:

    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()

        self.cat_features = [
            "Sex", "Drug", "Ascites",
            "Hepatomegaly", "Spiders", "Edema"
        ]

        self.num_features = [
            "Age", "N_Days", "Bilirubin",
            "Cholesterol", "Albumin", "Copper",
            "Alk_Phos", "SGOT", "Tryglicerides",
            "Platelets", "Prothrombin", "Stage"
        ]

    def preprocess(self, df, training=True):
        df = df.copy()

        if training:
            y = df["Status"]
            df = df.drop(columns=["Status", "id"])
            y = self.label_encoder.fit_transform(y)
        else:
            df = df.drop(columns=["id"])
            y = None

        for col in self.cat_features:
            df[col] = df[col].astype(str).fillna("Missing")

        for col in self.num_features:
            df[col] = df[col].fillna(df[col].median())

        return df, y

    def train_baseline(self, X, y):
        logging.info("Training RandomForest baseline...")

        X_encoded = X.copy()
        for col in self.cat_features:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col])

        rf = RandomForestClassifier(
            n_estimators=300,
            random_state=42
        )

        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []

        for train_idx, val_idx in kf.split(X_encoded, y):
            rf.fit(X_encoded.iloc[train_idx], y[train_idx])
            preds = rf.predict_proba(X_encoded.iloc[val_idx])
            score = log_loss(y[val_idx], preds)
            scores.append(score)

        logging.info(f"Baseline CV LogLoss: {np.mean(scores):.4f}")
        print("Baseline LogLoss:", np.mean(scores))

    def optimize_catboost(self, X, y):
        logging.info("Starting Optuna optimization...")

        def objective(trial):
            params = {
                "iterations": trial.suggest_int("iterations", 500, 1500),
                "depth": trial.suggest_int("depth", 4, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
                "loss_function": "MultiClass",
                "verbose": 0
            }

            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = []

            for train_idx, val_idx in kf.split(X, y):
                model = CatBoostClassifier(**params)
                model.fit(
                    X.iloc[train_idx],
                    y[train_idx],
                    cat_features=self.cat_features
                )
                preds = model.predict_proba(X.iloc[val_idx])
                scores.append(log_loss(y[val_idx], preds))

            return np.mean(scores)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=10)

        logging.info(f"Best params: {study.best_params}")
        print("Best params:", study.best_params)

        return study.best_params

    def train(self, dataset_path):
        logging.info("Training started...")
        df = pd.read_csv(dataset_path)

        X, y = self.preprocess(df, training=True)

        self.train_baseline(X, y)

        best_params = self.optimize_catboost(X, y)

        self.model = CatBoostClassifier(
            **best_params,
            loss_function="MultiClass",
            verbose=0
        )
        self.model.fit(X, y, cat_features=self.cat_features)

        os.makedirs("./model", exist_ok=True)
        joblib.dump(self.model, "./model/catboost_model.pkl")
        joblib.dump(self.label_encoder, "./model/label_encoder.pkl")

        logging.info("Training completed and model saved.")
        print("Model trained and saved!")

    def predict(self, dataset_path):
        logging.info("Prediction started...")

        df = pd.read_csv(dataset_path)
        ids = df["id"]

        X, _ = self.preprocess(df, training=False)

        model = joblib.load("./model/catboost_model.pkl")
        le = joblib.load("./model/label_encoder.pkl")

        preds = model.predict_proba(X)

        preds_df = pd.DataFrame(preds, columns=le.classes_)
        preds_df.insert(0, "id", ids)
        preds_df.columns = ["id", "Status_C", "Status_CL", "Status_D"]

        preds_df.to_csv("./data/results.csv", index=False)
        logging.info("Prediction completed.")
        print("Results saved to ./data/results.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train", "predict"])
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    model = My_Classifier_Model()

    if args.command == "train":
        model.train(args.dataset)
    elif args.command == "predict":
        model.predict(args.dataset)