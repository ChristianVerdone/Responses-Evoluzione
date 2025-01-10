import logging
import pandas as pd

_logger = logging.getLogger(__name__)


def load_and_process_data(file_paths, file_format="csv"):
    """
    Carica i dati dai percorsi specificati e li elabora (aggiunta colonna is_red e concatenazione).

    Args:
        file_paths (list): Lista dei percorsi dei file da caricare.
        file_format (str): Formato dei file (default: "csv").

    Returns:
        pandas.DataFrame: DataFrame contenente i dati aggregati.
    """
    if file_format == "csv":
        _logger.info("Caricamento dati CSV...")
        data_white = pd.read_csv(file_paths[0], delimiter=';')
        data_red = pd.read_csv(file_paths[1], delimiter=';')
        data_white["is_red"] = 0
        data_red["is_red"] = 1
        concatenated_df = pd.concat([data_white, data_red], ignore_index=True)
        return concatenated_df
    else:
        raise NotImplementedError(f"Formato file '{file_format}' non supportato.")
