from sklearn.ensemble import RandomForestClassifier


def train_model(X_train, y_train):
    """
  Addestra un modello RandomForestClassifier.

  Args:
      X_train (pd.DataFrame): Caratteristiche di training.
      y_train (pd.Series): Target di training.

  Returns:
      RandomForestClassifier: Modello addestrato.
  """

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model
