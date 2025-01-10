import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
import mlflow
import logging

# Configura il logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def evaluate_model(model, X_test, y_test):
    """
    Valuta il modello.

    Parametri:
    - model: Modello addestrato.
    - X_test (DataFrame): Dati di test.
    - y_test (Series): Target di test.

    Restituisce:
    - f1, precision, recall: Metriche di valutazione.
    """
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    return f1, precision, recall

if __name__ == "__main__":
    X_test = pd.read_csv("./data/X_test_transformed.csv")
    y_test = pd.read_csv("./data/y_test.csv").squeeze()

    model = mlflow.sklearn.load_model("model")

    with mlflow.start_run():
        f1, precision, recall = evaluate_model(model, X_test, y_test)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision_score", precision)
        mlflow.log_metric("recall_score", recall)

    logger.info(f"F1 Score: {f1}, Precision Score: {precision}, Recall Score: {recall}")
