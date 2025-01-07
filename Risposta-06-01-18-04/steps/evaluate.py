from sklearn.metrics import f1_score, precision_score, recall_score


def evaluate_model(model, X_test, y_test):
    """
  Valuta il modello utilizzando metriche di classificazione.

  Args:
      model (RandomForestClassifier): Modello da valutare.
      X_test (pd.DataFrame): Caratteristiche di test.
      y_test (pd.Series): Target di test.

  Returns:
      tuple: (F1 score, Precision score, Recall score)
  """

    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    return f1, precision, recall
