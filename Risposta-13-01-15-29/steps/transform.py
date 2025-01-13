import pandas as pd
import logging
from typing import DataFrame

logger = logging.getLogger(__name__)

def transformer_fn():
    """Returns a transformer object with fit and transform methods."""
    
    class CustomTransformer:
        def fit(self, X: DataFrame, y=None):
            return self
            
        def transform(self, X: DataFrame) -> DataFrame:
            logger.info("Applying transformations...")
            return X.dropna()
    
    return CustomTransformer()