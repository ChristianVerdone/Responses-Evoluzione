import logging
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
import mlflow

# Configura il logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    return f1, precision, recall

if __name__ == "__main__":
    import argparse
    import joblib

    parser = argparse.ArgumentParser(description="Step di valutazione per la recipe di MLflow")
    parser.add_argument("--model_path", required=True, help="Percorso del file del modello")
    parser.add_argument("--X_test_path", required=True, help="Percorso del file X_test")
    parser.add_argument("--y_test_path", required=True, help="Percorso del file y_test")
    args = parser.parse_args()

    # Carica il modello
    model = joblib.load(args.model_path)

    # Carica i dati di test
    X_test = pd.read_csv(args.X_test_path)
    y_test = pd.read_csv(args.y_test_path)

    # Valuta il modello
    f1, precision, recall = evaluate_model(model, X_test, y_test)

    # Avvia un run di MLflow
    with mlflow.start_run():
        # Log dei parametri e delle metriche
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision_score", precision)
        mlflow.log_metric("recall_score", recall)

        # Output delle metriche
        print("F1 Score:", f1)
        print("Precision Score:", precision)
        print("Recall Score:", recall)

        logger.info("Metriche di valutazione loggate con MLflow")
