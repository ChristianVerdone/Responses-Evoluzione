
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split

from ingest import ingest_data
from transform import transform_data
from train import train_model
from evaluate import evaluate_model

# Configurazione di MLflow
mlflow.set_tracking_uri("sqlite:///metadata/mlflow/mlruns.db")
mlflow.set_experiment("sklearn_classification_experiment")

# Ingestione dei dati
file_paths = ["./data/winequality-white.csv", "./data/winequality-red.csv"]
data = ingest_data(file_paths)

# Separazione in caratteristiche e target
target_col = "is_red"
X = data.drop(columns=[target_col])
y = data[target_col]

# Divisione dei dati (usando le SPLIT_RATIOS definite in local.yaml)
split_ratios = [0.80, 0.10, 0.10]
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=1 - split_ratios[0], random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp,
                                                test_size=split_ratios[2] / (split_ratios[1] + split_ratios[2]),
                                                random_state=42)

# Trasformazione dei dati
X_train = transform_data(X_train)
X_val = transform_data(X_val)
X_test = transform_data(X_test)

# Avvia un run di MLflow
with mlflow.start_run():

    # Addestramento del modello
    model = train_model(X_train, y_train)

    # Log del modello con MLflow
    mlflow.sklearn.log_model(model, "model")

    # Valutazione del modello
    f1, precision, recall = evaluate_model(model, X_test, y_test)

    # Log dei parametri e delle metriche
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision_score", precision)
    mlflow.log_metric("recall_score", recall)

    # Output delle metriche
    print("F1 Score:", f1)
    print("Precision Score:", precision)
    print("Recall Score:", recall)

print("Esperimento completato!")
