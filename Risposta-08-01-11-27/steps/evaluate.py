import pandas as pd
import joblib
from sklearn.metrics import f1_score, precision_score, recall_score
import logging

logger = logging.getLogger("EvaluateStep")
logger.setLevel(logging.INFO)

def evaluate_model(model, X_test, y_test):
    """
    Valuta il modello calcolando F1, precision e recall.
    """
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    return f1, precision, recall

if __name__ == "__main__":
    # Carica i dati trasformati
    data = pd.read_csv("./data/transformed_data.csv")
    target_col = "is_red"
    X = data.drop(columns=[target_col])
    y = data[target_col]

    # Carica il modello
    model = joblib.load("./models/trained_model.pkl")

    # Valuta il modello
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    f1, precision, recall = evaluate_model(model, X_test, y_test)

    # Salva le metriche
    metrics = {"f1_score": f1, "precision_score": precision, "recall_score": recall}
    logger.info("Metriche: %s", metrics)
    pd.DataFrame([metrics]).to_csv("./metrics/evaluation_metrics.csv", index=False)
    logger.info("Metriche salvate in ./metrics/evaluation_metrics.csv.")
