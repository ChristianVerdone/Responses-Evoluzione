from sklearn.ensemble import RandomForestClassifier
import mlflow
import logging
from mlflow.pyfunc import PythonModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainStep(PythonModel):
    def __init__(self, config):
        self.config = config

    def predict(self, context, model_input):
        X_train, y_train = model_input
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        mlflow.log_param("model_type", "RandomForest")
        mlflow.sklearn.log_model(model, "model")
        logger.info("Modello addestrato e registrato")
        return model