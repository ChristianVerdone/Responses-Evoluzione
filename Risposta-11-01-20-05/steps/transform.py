import logging
from typing import Dict, Any
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WineQualityTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer for wine quality data."""
    
    def __init__(self, params: Dict[str, Any] = None):
        self.params = params or {}
        
    def fit(self, X: pd.DataFrame, y=None):
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the wine quality data."""
        logger.info("Transforming data")
        X_transformed = X.copy()
        
        # Drop NA values if specified
        if self.params.get('drop_na', True):
            X_transformed = X_transformed.dropna()
            logger.info(f"Dropped NA values. New shape: {X_transformed.shape}")
            
        return X_transformed

def transformer_fn(params: Dict[str, Any] = None) -> WineQualityTransformer:
    """Create and return the wine quality transformer."""
    return WineQualityTransformer(params)