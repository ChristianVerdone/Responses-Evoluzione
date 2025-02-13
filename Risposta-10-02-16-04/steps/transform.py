import logging
from mlflow.pyfunc import PythonModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def transform_data(data):
    return data.dropna()

class TransformStep(PythonModel):
    def __init__(self, config):
        self.config = config

    def predict(self, context, model_input):
        X_train, X_val, X_test = model_input
        return (
            transform_data(X_train),
            transform_data(X_val),
            transform_data(X_test)
        )