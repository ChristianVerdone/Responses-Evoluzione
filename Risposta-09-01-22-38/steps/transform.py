def transform_data(data, params):
    """
    Trasforma il dataset applicando operazioni specificate nei parametri.
    
    :param data: DataFrame da trasformare.
    :param params: Parametri di trasformazione.
    :return: DataFrame trasformato.
    """
    if params.get("dropna", False):
        data = data.dropna()
    return data