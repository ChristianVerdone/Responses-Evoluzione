from sklearn.metrics import f1_score, precision_score, recall_score


def evaluate_model(model, X_test, y_test):
    """Valuta il modello e restituisce le metriche di performance.

  Args:
      model (sklearn.ensemble.RandomForestClassifier): Modello addestrato.
      X_test (pandas.DataFrame): Caratteristiche di test.
      y_test (pandas.Series): Target di test.

  Returns:
      tuple: (float, float, float) - (F1-score, Precision, Recall)
  """
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    return f1, precision, recall
