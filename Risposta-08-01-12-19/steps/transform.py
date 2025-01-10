# steps/transform.py
def transform_data(data):
    """
    Applica le trasformazioni necessarie al dataset.
    
    Args:
        data: DataFrame di input
    Returns:
        DataFrame: Dataset trasformato
    """
    return data.dropna()

def transformer_fn():
    """
    Returns:
        Transformer object con metodi fit e transform
    """
    class CustomTransformer:
        def fit(self, X, y=None):
            return self
        
        def transform(self, X):
            return transform_data(X)
    
    return CustomTransformer()