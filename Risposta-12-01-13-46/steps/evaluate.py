from sklearn.metrics import f1_score, precision_score, recall_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_model(model, X_test, y_test):
    """
    Valuta il modello sui dati di test.
    Args:
        model: Modello addestrato.
        X_test: Feature del test set.
        y_test: Target del test set.
    Returns:
        dict: Metriche di valutazione.
    """
    logger.info("Valutazione del modello...")
    y_pred = model.predict(X_test)

    metrics = {
        "f1_score": f1_score(y_test, y_pred),
        "precision_score": precision_score(y_test, y_pred),
        "recall_score": recall_score(y_test, y_pred)
    }

    logger.info(f"Metriche: {metrics}")
    return metrics
