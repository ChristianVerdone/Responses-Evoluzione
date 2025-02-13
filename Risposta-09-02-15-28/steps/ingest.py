import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ingest_data(file_paths, delimiter=";"):
    """Carica e unisce i dataset."""
    logger.info("Caricamento dati...")
    data_white = pd.read_csv(file_paths[0], delimiter=delimiter)
    data_red = pd.read_csv(file_paths[1], delimiter=delimiter)

    data_white["is_red"] = 0
    data_red["is_red"] = 1

    data = pd.concat([data_white, data_red], ignore_index=True)
    return data


def ingest_step():
    from jinja2 import Environment, FileSystemLoader
    env = Environment(loader=FileSystemLoader("."))

    # Carica i parametri da local.yaml
    location = ["{{INGEST_CONFIG.location}}"]  # Jinja2 placeholder
    delimiter = "{{INGEST_CONFIG.delimiter}}"

    return ingest_data(location, delimiter)