import logging
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

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
    logger.info("Addestramento del modello RandomForestClassifier")
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Valuta il modello addestrato.

    :param model: Modello addestrato
    :param X_test: DataFrame contenente le caratteristiche di test
    :param y_test: Series contenente il target di test
    :return: Tuple contenente f1_score, precision_score e recall_score
    """
    logger.info("Valutazione del modello")
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    return f1, precision, recall


# Esempio di utilizzo della funzione
if __name__ == "__main__":
    mlflow.set_tracking_uri("sqlite:///metadata/mlflow/mlruns.db")
    mlflow.set_experiment("sklearn_classification_experiment")

    X_train = pd.read_csv("/data/X_train_transformed.csv")
    y_train = pd.read_csv("/data/y_train.csv").squeeze()
    X_test = pd.read_csv("/data/X_test_transformed.csv")
    y_test = pd.read_csv("/data/y_test.csv").squeeze()

    with mlflow.start_run():
        model = train_model(X_train, y_train)
        mlflow.sklearn.log_model(model, "model")

        f1, precision, recall = evaluate_model(model, X_test, y_test)

        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision_score", precision)
        mlflow.log_metric("recall_score", recall)

        print("F1 Score:", f1)
        print("Precision Score:", precision)
        print("Recall Score:", recall)
