from sklearn.metrics import f1_score, precision_score, recall_score


def evaluate_model(model, X_test, y_test):
    """
    Valuta il modello con f1_score, precision_score e recall_score.

    Args:
        model: Modello addestrato.
        X_test: Feature del test set.
        y_test: Target del test set.

    Returns:
        tuple: Una tupla contenente f1_score, precision_score e recall_score.
    """
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    return f1, precision, recall
