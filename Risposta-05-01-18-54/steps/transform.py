import pandas as pd
import logging

# Configura il logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def transform_data(data):
    """
    Esegue trasformazioni personalizzate sui dati.

    :param data: DataFrame contenente i dati da trasformare
    :return: DataFrame trasformato
    """
    data = data.dropna()
    return data


if __name__ == "__main__":
    data = pd.read_csv("./data/ingested_data.csv")
    transformed_data = transform_data(data)
    transformed_data.to_csv("./data/transformed_data.csv", index=False)