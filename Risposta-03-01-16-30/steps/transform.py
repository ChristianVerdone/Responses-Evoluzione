def transformer_fn():
    class CustomTransformer:
        def fit(self, X, y=None):
            return self
            
        def transform(self, X):
            return X.dropna()
    
    return CustomTransformer()