import pandas as pd
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
import joblib

# Configura il logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def train_model(X_train, y_train):
    logger.info("Addestramento del modello...")
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    logger.info("Addestramento completato.")
    return model

def evaluate_model(model, X_test, y_test):
    logger.info("Valutazione del modello...")
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    return f1, precision, recall

if __name__ == "__main__":
    data = pd.read_csv("data/transformed_data.csv")
    target_col = "is_red"
    X = data.drop(columns=[target_col])
    y = data[target_col]

    # Divisione in train/test/val
    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    model = train_model(X_train, y_train)
    joblib.dump(model, "models/model.pkl")
    f1, precision, recall = evaluate_model(model, X_test, y_test)
    logger.info(f"F1 Score: {f1}, Precision: {precision}, Recall: {recall}")
