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

from clearml import Task


# ---------------------------
# Singleton Logger
# ---------------------------
def get_logger():
    logger = logging.getLogger("liver_cirrhosis_model")
    if not logger.handlers:
        os.makedirs("./data", exist_ok=True)
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        )
        file_handler = logging.FileHandler("./data/log_file.log")
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    return logger


class My_Classifier_Model:

    def __init__(self, mode):

        self.logger_py = get_logger()

        self.task = None
        self.logger = None

        if mode == "train":
            try:
                self.task = Task.init(
                    project_name="LiverCirrhosis",
                    task_name="CatBoost_Training",
                    output_uri=None
                )
                self.logger = self.task.get_logger()

            except Exception:
                self.logger_py.exception("ClearML initialization failed")

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

    # ---------------------------
    # Preprocessing
    # ---------------------------
    def preprocess(self, df, training=True):
        try:
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
        except Exception:
            self.logger_py.exception("Preprocessing failed")
            raise

    # ---------------------------
    # Baseline model
    # ---------------------------
    def train_baseline(self, X, y):
        try:
            self.logger_py.info("Training RandomForest baseline...")
            X_encoded = X.copy()
            for col in self.cat_features:
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X_encoded[col])

            rf = RandomForestClassifier(n_estimators=300, random_state=42)
            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = []

            for fold, (train_idx, val_idx) in enumerate(kf.split(X_encoded, y), 1):
                rf.fit(X_encoded.iloc[train_idx], y[train_idx])
                preds = rf.predict_proba(X_encoded.iloc[val_idx])
                score = log_loss(y[val_idx], preds)
                scores.append(score)
                if self.logger:
                    self.logger.report_scalar("Baseline LogLoss", "fold", score, iteration=fold)

            mean_score = np.mean(scores)
            self.logger_py.info(f"Baseline CV LogLoss: {mean_score:.4f}")
            if self.logger:
                self.logger.report_text(f"Baseline CV LogLoss: {mean_score:.4f}")
            print("Baseline LogLoss:", mean_score)
        except Exception:
            self.logger_py.exception("Baseline training failed")

    # ---------------------------
    # Optuna optimization
    # ---------------------------
    def optimize_catboost(self, X, y):
        try:
            self.logger_py.info("Starting Optuna optimization...")
            if isinstance(y, pd.Series):
                y = y.values

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
                    model.fit(X.iloc[train_idx], y[train_idx], cat_features=self.cat_features)
                    preds = model.predict_proba(X.iloc[val_idx])
                    scores.append(log_loss(y[val_idx], preds))
                return np.mean(scores)

            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=10)

            self.logger_py.info(f"Best params: {study.best_params}")
            print("Best params:", study.best_params)

            # ---------------------------
            # Log Best Model Params to ClearML
            # ---------------------------
            if self.logger:
                self.logger.report_text(f"Best params: {study.best_params}")

            return study.best_params
        except Exception:
            self.logger_py.exception("Optuna optimization failed")
            raise

    # ---------------------------
    # Training pipeline
    # ---------------------------
    def train(self, dataset_path):
        try:
            self.logger_py.info("Training started...")

            try:
                df = pd.read_csv(dataset_path)
            except FileNotFoundError:
                self.logger_py.exception(f"Dataset not found: {dataset_path}")
                raise

            # ---------------------------
            # Upload train dataset to ClearML
            # ---------------------------
            if self.task:
                self.task.upload_artifact(name="train_dataset", artifact_object=dataset_path)

            X, y = self.preprocess(df, training=True)
            self.train_baseline(X, y)
            best_params = self.optimize_catboost(X, y)

            self.model = CatBoostClassifier(**best_params, loss_function="MultiClass", verbose=0)

            try:
                self.model.fit(X, y, cat_features=self.cat_features)
            except Exception:
                self.logger_py.exception("Model training failed")
                raise

            preds = self.model.predict_proba(X)
            final_logloss = log_loss(y, preds)

            if self.logger:
                self.logger.report_scalar("Final LogLoss", "all_data", final_logloss, iteration=1)

            try:
                os.makedirs("./model", exist_ok=True)
                joblib.dump(self.model, "./model/catboost_model.pkl")
                joblib.dump(self.label_encoder, "./model/label_encoder.pkl")
            except Exception:
                self.logger_py.exception("Model saving failed")
                raise

            if self.task:
                self.task.upload_artifact(name="catboost_model", artifact_object=self.model)
                self.task.upload_artifact(name="label_encoder", artifact_object=self.label_encoder)

            self.logger_py.info("Training completed and model saved.")
            print("Model trained and saved!")

        except Exception:
            self.logger_py.exception("Training pipeline failed")
            raise

    # ---------------------------
    # Prediction pipeline
    # ---------------------------
    def predict(self, dataset_path):
        try:
            self.logger_py.info("Prediction started...")

            try:
                df = pd.read_csv(dataset_path)
            except Exception:
                self.logger_py.exception("Prediction dataset loading failed")
                raise

            ids = df["id"]
            X, _ = self.preprocess(df, training=False)

            try:
                model = joblib.load("./model/catboost_model.pkl")
                le = joblib.load("./model/label_encoder.pkl")
            except Exception:
                self.logger_py.exception("Failed to load trained model")
                raise

            preds = model.predict_proba(X)
            preds_df = pd.DataFrame(preds, columns=le.classes_)
            preds_df.insert(0, "id", ids)
            preds_df.columns = ["id", "Status_C", "Status_CL", "Status_D"]

            try:
                preds_df.to_csv("./data/results.csv", index=False)
            except Exception:
                self.logger_py.exception("Failed to save prediction results")
                raise

            if self.logger:
                self.logger.report_text("Prediction saved to ./data/results.csv")

            self.logger_py.info("Prediction completed.")
            print("Results saved to ./data/results.csv")

        except Exception:
            self.logger_py.exception("Prediction pipeline failed")
            raise


# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    logger = get_logger()
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("command", choices=["train", "predict"])
        parser.add_argument("--dataset", required=True)
        args = parser.parse_args()

        model = My_Classifier_Model(args.command)
        if args.command == "train":
            model.train(args.dataset)
        elif args.command == "predict":
            model.predict(args.dataset)
    except Exception:
        logger.exception("Application crashed")
        raise