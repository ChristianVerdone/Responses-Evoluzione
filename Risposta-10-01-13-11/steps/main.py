### Main MLflow Script: main.py

import mlflow
import mlflow.sklearn
from ingest import ingest_data
from split import split_data
from transform import transform_data
from train import train_model
from evaluate import evaluate_model

file_paths = ["./data/winequality-white.csv", "./data/winequality-red.csv"]

data = ingest_data(file_paths)

split_ratios = [0.8, 0.1, 0.1]
X_train, X_val, X_test, y_train, y_val, y_test = split_data(data, "is_red", split_ratios)

X_train = transform_data(X_train)
X_val = transform_data(X_val)
X_test = transform_data(X_test)

with mlflow.start_run():
    model = train_model(X_train, y_train)
    mlflow.sklearn.log_model(model, "model")

    metrics = evaluate_model(model, X_test, y_test)

    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, metric_value)

    print(metrics)