from sklearn.metrics import f1_score, precision_score, recall_score


def evaluate_model(model, X_test, y_test):
    """Valuta un modello classificatore.

  Args:
      model: Modello da valutare.
      X_test: Matrice di features per la valutazione.
      y_test: Vettore di target per la valutazione.

  Returns:
      Tuple contenente F1-score, precision e recall.
  """

    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    return f1, precision, recall
