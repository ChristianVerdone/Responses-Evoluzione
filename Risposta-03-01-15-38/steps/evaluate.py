from sklearn.metrics import f1_score, precision_score, recall_score
import logging

# Configura il logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def evaluate_model(model, X_test, y_test):
    """
    Valuta il modello sui dati di test.

    :param model: Modello addestrato
    :param X_test: DataFrame delle caratteristiche di test
    :param y_test: Serie dei target di test
    :return: Tuple contenente f1_score, precision_score e recall_score
    """
    logger.info("Valutazione del modello.")
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    return f1, precision, recall

'''
# Esempio di utilizzo
f1, precision, recall = evaluate_model(model, X_test, y_test)'''