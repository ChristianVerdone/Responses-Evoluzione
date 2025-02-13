from sklearn.ensemble import RandomForestClassifier
import mlflow
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train(X_train, y_train, params):
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    mlflow.sklearn.log_model(model, "model")
    logger.info("Modello addestrato e registrato in MLflow")
    return model