from sklearn.ensemble import RandomForestClassifier


def train_model(X_train, y_train):
    """
    Addestra il modello di classificazione Random Forest.

    Args:
      X_train (pandas.DataFrame): DataFrame contenente le feature di training.
      y_train (pandas.Series): Series contenente le etichette di training.

    Returns:
      sklearn.ensemble.RandomForestClassifier: Modello addestrato.
    """
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model
