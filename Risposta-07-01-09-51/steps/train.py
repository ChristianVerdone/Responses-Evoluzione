import logging
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn

# Configura il logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def train_model(X_train, y_train):
    """
    Addestra un modello RandomForestClassifier.

    :param X_train: DataFrame contenente le caratteristiche di addestramento
    :param y_train: Series contenente il target di addestramento
    :return: Modello addestrato
    """
    logger.info("Addestramento del modello.")

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Valuta le prestazioni del modello.

    :param model: Modello addestrato
    :param X_test: DataFrame contenente le caratteristiche di test
    :param y_test: Series contenente il target di test
    :return: Tuple contenente f1_score, precision_score e recall_score
    """
    logger.info("Valutazione del modello.")

    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    logger.info("F1 Score: %f", f1)
    logger.info("Precision Score: %f", precision)
    logger.info("Recall Score: %f", recall)

    return f1, precision, recall


# Esempio di utilizzo delle funzioni definite sopra
data = pd.read_csv("./data/transformed_data.csv")

# Separazione in caratteristiche e target
target_col = "is_red"
X = data.drop(columns=[target_col])
y = data[target_col]

# Divisione dei dati (usando le SPLIT_RATIOS definite in local.yaml)
split_ratios = [0.80, 0.10, 0.10]
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=1 - split_ratios[0], random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp,
                                                test_size=split_ratios[2] / (split_ratios[1] + split_ratios[2]),
                                                random_state=42)

# Avvia un run di MLflow
with mlflow.start_run():
    # Addestramento del modello
    model = train_model(X_train, y_train)

    # Log del modello con MLflow
    mlflow.sklearn.log_model(model, "model")

    # Valutazione del modello
    f1, precision, recall = evaluate_model(model, X_test, y_test)

    # Log dei parametri e delle metriche
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision_score", precision)
    mlflow.log_metric("recall_score", recall)

    # Output delle metriche
    print("F1 Score:", f1)
    print("Precision Score:", precision)
    print("Recall Score:", recall)

print("Esperimento completato!")