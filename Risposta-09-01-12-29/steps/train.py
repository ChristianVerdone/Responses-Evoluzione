from sklearn.ensemble import RandomForestClassifier


def train_model(X_train, y_train):
    """Addestra un modello RandomForestClassifier.

  Args:
      X_train (pandas.DataFrame): Caratteristiche di training.
      y_train (pandas.Series): Target di training.

  Returns:
      sklearn.ensemble.RandomForestClassifier: Modello addestrato.
  """
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model
