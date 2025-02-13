from sklearn.metrics import f1_score, precision_score, recall_score
import logging

logger = logging.getLogger(__name__)


def evaluate_model(model, X_test, y_test):
    """
    Calcola le metriche di valutazione.
    """
    y_pred = model.predict(X_test)

    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    logger.info("Metriche: F1=%.2f, Precision=%.2f, Recall=%.2f", f1, precision, recall)
    return {"f1_score": f1, "precision_score": precision, "recall_score": recall}