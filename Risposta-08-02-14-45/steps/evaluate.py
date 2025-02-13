from sklearn.metrics import f1_score, precision_score, recall_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_model(model, X_test, y_test):
    """Calcola le metriche di valutazione."""
    y_pred = model.predict(X_test)
    return {
        "f1_score": f1_score(y_test, y_pred),
        "precision_score": precision_score(y_test, y_pred),
        "recall_score": recall_score(y_test, y_pred)
    }