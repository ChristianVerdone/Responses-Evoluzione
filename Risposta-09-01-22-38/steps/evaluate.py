from sklearn.metrics import f1_score, precision_score, recall_score

def evaluate_model(model, X_test, y_test, thresholds):
    """
    Valuta il modello sui dati di test.
    
    :param model: Modello da valutare.
    :param X_test: Dati di input di test.
    :param y_test: Etichette di test.
    :param thresholds: Soglie di valutazione.
    :return: Dizionario con le metriche calcolate e se il modello è valido.
    """
    y_pred = model.predict(X_test)

    metrics = {
        "f1_score": f1_score(y_test, y_pred),
        "precision_score": precision_score(y_test, y_pred),
        "recall_score": recall_score(y_test, y_pred),
    }

    is_valid = all(metrics[m] >= thresholds.get(m, 0) for m in thresholds)

    return metrics, is_valid