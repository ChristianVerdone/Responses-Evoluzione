from sklearn.ensemble import RandomForestClassifier


def train_model(X_train, y_train):
    """
    Addestra un modello RandomForest sui dati di training.

    :param X_train: Caratteristiche di training
    :param y_train: Target di training
    :return: Modello addestrato
    """
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model