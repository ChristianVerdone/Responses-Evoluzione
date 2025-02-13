import logging
from sklearn.metrics import f1_score, precision_score, recall_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_fn(model, X_test, y_test):
    """
    Valuta il modello e restituisce le metriche.
    """
    try:
        y_pred = model.predict(X_test)
        return {
            "f1_score": f1_score(y_test, y_pred),
            "precision_score": precision_score(y_test, y_pred),
            "recall_score": recall_score(y_test, y_pred)
        }
    except Exception as e:
        logger.error(f"Errore nella valutazione: {e}")
        raise