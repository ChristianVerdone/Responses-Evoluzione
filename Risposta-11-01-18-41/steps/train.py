import logging
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn

# Configura il logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Step di training per la recipe di MLflow")
    parser.add_argument("--X_train_path", required=True, help="Percorso del file X_train")
    parser.add_argument("--y_train_path", required=True, help="Percorso del file y_train")
    args = parser.parse_args()

    # Carica i dati di training
    X_train = pd.read_csv(args.X_train_path)
    y_train = pd.read_csv(args.y_train_path)

    # Addestra il modello
    model = train_model(X_train, y_train)

    # Avvia un run di MLflow
    with mlflow.start_run():
        # Log del modello con MLflow
        mlflow.sklearn.log_model(model, "model")
        logger.info("Modello addestrato e loggato con MLflow")
