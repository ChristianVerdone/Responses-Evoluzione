import pandas as pd
import logging
from mlflow.pyfunc import PythonModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ingest_data(file_paths):
    data_white = pd.read_csv(file_paths[0], delimiter=';')
    data_red = pd.read_csv(file_paths[1], delimiter=';')

    data_white["is_red"] = 0
    data_red["is_red"] = 1

    data = pd.concat([data_white, data_red], ignore_index=True)
    logger.info("Dati ingeriti con successo")
    return data


class IngestStep(PythonModel):
    def __init__(self, config):
        self.config = config

    def predict(self, context, model_input):
        return ingest_data(self.config['location'])