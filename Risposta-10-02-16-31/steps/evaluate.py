from sklearn.metrics import f1_score, precision_score, recall_score
import logging

# Configura il logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_model(model, X_test, y_test):
    """
    Valuta il modello utilizzando F1, Precision e Recall.
    """
    logger.info("Valutazione del modello...")
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    logger.info(f"F1 Score: {f1}, Precision: {precision}, Recall: {recall}")
    return f1, precision, recall