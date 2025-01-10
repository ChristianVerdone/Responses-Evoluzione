import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger("TrainStep")
logger.setLevel(logging.INFO)

def train_model(X_train, y_train):
    """
    Addestra il modello RandomForestClassifier.
    """
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    # Carica i dati trasformati
    data = pd.read_csv("./data/transformed_data.csv")
    target_col = "is_red"
    X = data.drop(columns=[target_col])
    y = data[target_col]

    # Divisione dei dati
    split_ratios = [0.80, 0.10, 0.10]
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=1 - split_ratios[0], random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=split_ratios[2] / (split_ratios[1] + split_ratios[2]),
        random_state=42
    )

    # Addestra il modello
    model = train_model(X_train, y_train)

    # Salva il modello
    joblib.dump(model, "./models/trained_model.pkl")
    logger.info("Modello salvato in ./models/trained_model.pkl.")
