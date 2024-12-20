from sklearn.metrics import f1_score, precision_score, recall_score


def evaluate_model(model, X_test, y_test):
    """
    Valuta il modello sui dati di test.

    :param model: Modello addestrato
    :param X_test: Caratteristiche di test
    :param y_test: Target di test
    :return: Tuple contenente f1_score, precision_score e recall_score
    """
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    return f1, precision, recall