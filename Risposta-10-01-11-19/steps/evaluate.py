from sklearn.ensemble import RandomForestClassifier

def train_model(X_train, y_train, model_params):
    """
    Addestra un modello RandomForestClassifier.
    """
    model = RandomForestClassifier(**model_params)
    model.fit(X_train, y_train)
    return model
