# Liver Cirrhosis Survival Prediction

## Author

**Full Name:** Гранин Егор Вадимович
**Group:** 972401

---

# Project Description

This project implements a machine learning pipeline for predicting the survival status of patients with liver cirrhosis.

The model is based on **CatBoost Gradient Boosting** and uses automated hyperparameter tuning with **Optuna**.
Experiment tracking is implemented using **ClearML**.

The pipeline includes:

* Data preprocessing
* Baseline model (RandomForest)
* Hyperparameter optimization
* Model training
* Model evaluation
* Prediction generation
* Experiment logging

---

# Project Structure

```
LiverCirrhosisModel/
│
├── liver_cirrhosis_model/
│   └── model.py
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── model/
│   ├── catboost_model.pkl
│   └── label_encoder.pkl
│
├── Dockerfile
├── pyproject.toml
├── poetry.lock
└── README.md
```

---

# Requirements

* Python 3.12
* Poetry
* Docker (optional)

Main libraries:

* pandas
* numpy
* scikit-learn
* CatBoost
* Optuna
* ClearML

---

# Installation (Without Docker)

Clone the repository:

```
git clone <repo_url>
cd LiverCirrhosisModel
```

Install dependencies using Poetry:

```
poetry install
```

Activate environment:

```
poetry shell
```

---

# Train Model (Without Docker)

Place dataset in:

```
data/train.csv
```

Run training:

```
python liver_cirrhosis_model/model.py train --dataset data/train.csv
```

This will:

* preprocess data
* train baseline model
* run Optuna optimization
* train CatBoost model
* save model to

```
model/catboost_model.pkl
```

---

# Run Prediction (Without Docker)

Place test dataset:

```
data/test.csv
```

Run:

```
python liver_cirrhosis_model/model.py predict --dataset data/test.csv
```

Results will be saved to:

```
data/results.csv
```

---

# Using ClearML

To enable experiment tracking with ClearML:

Create API credentials at:

```
https://app.clear.ml
```

Then run:

```
clearml-init
```

Follow the prompts to configure API access.

After configuration, all experiments will appear in the ClearML dashboard.

---

# Docker Usage

## Build Docker Image

```
docker build -t liver_model .
```

---

## Train Model in Docker

Make sure dataset exists:

```
data/train.csv
```

Run container:

```
docker run -v $(pwd)/data:/app/data -e CLEARML_WEB_HOST=https://app.clear.ml -e CLEARML_API_HOST=https://api.clear.ml -e CLEARML_FILES_HOST=https://files.clear.ml -e CLEARML_API_ACCESS_KEY={ACCESS_KEY} -e CLEARML_API_SECRET_KEY={SECRET_KEY} liver_model
```

This mounts your local `data` folder into the container.

---

## Run Prediction in Docker

```
docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/model:/app/model liver_model predict --dataset /app/data/test.csv
```

---


# Evaluation Metric

The main evaluation metric used is:

Log Loss

Cross-validation strategy:

StratifiedKFold with 5 folds.

