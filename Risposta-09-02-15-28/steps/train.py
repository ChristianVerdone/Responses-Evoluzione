from sklearn.ensemble import RandomForestClassifier
import mlflow


def train_model(X_train, y_train, estimator_params=None):
    """Addestra un modello RandomForest."""
    model = RandomForestClassifier(**(estimator_params or {}))
    model.fit(X_train, y_train)

    # Log del modello con MLflow
    mlflow.sklearn.log_model(model, "model")
    return model


def train_step(X_train, y_train):
    estimator_params = {{TRAIN_CONFIG.estimator_params}}  # Valore da Jinja2
    return train_model(X_train, y_train, estimator_params)