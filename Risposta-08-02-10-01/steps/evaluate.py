from sklearn.metrics import f1_score, precision_score, recall_score
import mlflow
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate(model, X_test, y_test, thresholds):
    y_pred = model.predict(X_test)

    metrics = {
        "f1_score": f1_score(y_test, y_pred),
        "precision_score": precision_score(y_test, y_pred),
        "recall_score": recall_score(y_test, y_pred)
    }

    # Log delle metriche in MLflow
    for name, value in metrics.items():
        mlflow.log_metric(name, value)
        logger.info("%s: %.2f", name, value)

    # Verifica delle soglie
    if not all(metrics[k] >= thresholds[k] for k in thresholds):
        logger.warning("Il modello non soddisfa tutte le soglie di validazione!")
        return False
    return True