from sklearn.model_selection import train_test_split
import logging
from mlflow.pyfunc import PythonModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SplitStep(PythonModel):
    def __init__(self, config):
        self.config = config

    def predict(self, context, model_input):
        X = model_input.drop(columns=[self.config['target_col']])
        y = model_input[self.config['target_col']]

        ratios = self.config['split_ratios']
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=1 - ratios[0], random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=ratios[2] / (ratios[1] + ratios[2]),
            random_state=42
        )
        logger.info("Split dati completato")
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)