from sklearn.ensemble import RandomForestClassifier

def train_model(X_train, y_train, params):
    """
    Addestra un modello RandomForest.
    
    :param X_train: Dati di input per il training.
    :param y_train: Target per il training.
    :param params: Parametri del modello.
    :return: Modello addestrato.
    """
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model