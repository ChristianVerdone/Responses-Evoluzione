from sklearn.model_selection import train_test_split
import logging

# Configura il logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def split_data(X, y, split_ratios):
    """
    Divide i dati in training, validation e test set.
    """
    logger.info(f"Dividendo i dati con rapporti: {split_ratios}")
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=1 - split_ratios[0], random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp,
                                                    test_size=split_ratios[2] / (split_ratios[1] + split_ratios[2]),
                                                    random_state=42)
    logger.info("Dati divisi correttamente.")
    return X_train, X_val, X_test, y_train, y_val, y_test