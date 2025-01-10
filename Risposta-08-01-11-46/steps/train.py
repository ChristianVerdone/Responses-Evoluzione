import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
import logging

# Configura il logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def train_model(X_train, y_train):
    """
    Addestra il modello RandomForestClassifier.

    Parametri:
    - X_train (DataFrame): Dati di training.
    - y_train (Series): Target di training.

    Restituisce:
    - model: Modello addestrato.
    """
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    X_train = pd.read_csv("./data/X_train_transformed.csv")
    y_train = pd.read_csv("./data/y_train.csv").squeeze()

    with mlflow.start_run():
        model = train_model(X_train, y_train)
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_param("model_type", "RandomForest")

    logger.info("Modello addestrato e registrato con MLflow")
