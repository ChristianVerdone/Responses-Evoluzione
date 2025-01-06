from sklearn.ensemble import RandomForestClassifier


def train_model(X_train, y_train):
    """Addestra un modello Random Forest Classifier.

  Args:
      X_train: Matrice di features per l'addestramento.
      y_train: Vettore di target per l'addestramento.

  Returns:
      Un modello RandomForestClassifier addestrato.
  """

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model
