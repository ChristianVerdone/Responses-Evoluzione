from sklearn.ensemble import RandomForestClassifier


def estimator_fn(estimator_params=None):
    """
    Definisce il modello RandomForest per l'addestramento.
    """
    if estimator_params is None:
        estimator_params = {}

    model = RandomForestClassifier(**estimator_params)
    return model