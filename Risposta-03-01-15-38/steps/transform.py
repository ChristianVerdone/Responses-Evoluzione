import pandas as pd
import logging

# Configura il logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def transform_data(data):
    """
    Esegue trasformazioni personalizzate sui dati.

    :param data: DataFrame contenente i dati
    :return: DataFrame trasformato
    """
    logger.info("Trasformazione dei dati.")
    data = data.dropna()
    return data

'''
# Esempio di utilizzo
data = transform_data(data)'''