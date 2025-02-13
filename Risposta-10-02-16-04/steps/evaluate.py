from sklearn.metrics import f1_score, precision_score, recall_score
import logging
from mlflow.pyfunc import PythonModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValidateStep(PythonModel):
    def __init__(self, config):
        self.config = config

    def predict(self, context, model_input):
        model, (X_test, y_test) = model_input
        y_pred = model.predict(X_test)

        metrics = {
            "f1_score": f1_score(y_test, y_pred),
            "precision_score": precision_score(y_test, y_pred),
            "recall_score": recall_score(y_test, y_pred)
        }

        validation_status = all([
            metrics["f1_score"] >= self.config['thresholds']['f1_score'],
            metrics["precision_score"] >= self.config['thresholds']['precision_score'],
            metrics["recall_score"] >= self.config['thresholds']['recall_score']
        ])

        if validation_status or self.config['allow_non_validated_model']:
            logger.info("Modello validato con successo")
        else:
            logger.warning("Modello non soddisfa i criteri di validazione")

        return metrics