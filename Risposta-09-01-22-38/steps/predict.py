def predict_model(model, X_input):
    """
    Esegue le predizioni con il modello fornito.
    
    :param model: Modello addestrato.
    :param X_input: Dati di input.
    :return: Predizioni.
    """
    return model.predict(X_input)