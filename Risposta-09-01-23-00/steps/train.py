import logging
from sklearn.ensemble import RandomForestClassifier

# Configura il logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def train_model(X_train, y_train):
    """
    Addestra il modello.

    Parametri:
    - X_train (pd.DataFrame): Dati di training.
    - y_train (pd.Series): Etichette di training.

    Restituisce:
    - RandomForestClassifier: Modello addestrato.
    """
    logger.info("Inizio dell'addestramento del modello...")
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    logger.info("Addestramento del modello completato.")
    return model

if __name__ == "__main__":
    # Esempio di DataFrame e Series
    X_train = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature2': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    })
    y_train = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    model = train_model(X_train, y_train)
    print("Modello addestrato:", model)
