import pandas as pd
import logging

logger = logging.getLogger(__name__)


def ingest_data(location, delimiter=";"):
    """
    Carica e unisce i dataset dei vini bianchi e rossi.
    """
    data_white = pd.read_csv(location[0], delimiter=delimiter)
    data_red = pd.read_csv(location[1], delimiter=delimiter)

    data_white["is_red"] = 0
    data_red["is_red"] = 1

    data = pd.concat([data_white, data_red], ignore_index=True)
    logger.info("Dati caricati correttamente. Dimensioni: %s", data.shape)
    return data