import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
import logging

# Configura il logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def train_model(X_train, y_train):
    """
    Addestra un modello di RandomForestClassifier.

    :param X_train: DataFrame contenente le caratteristiche di training
    :param y_train: Series contenente i target di training
    :return: Modello addestrato
    """
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Valuta il modello addestrato.

    :param model: Modello addestrato
    :param X_test: DataFrame contenente le caratteristiche di test
    :param y_test: Series contenente i target di test
    :return: Tuple contenente f1_score, precision_score e recall_score
    """
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    return f1, precision, recall


if __name__ == "__main__":
    mlflow.set_tracking_uri("sqlite:///metadata/mlflow/mlruns.db")
    mlflow.set_experiment("sklearn_classification_experiment")

    X_train = pd.read_csv("./data/X_train.csv")
    y_train = pd.read_csv("./data/y_train.csv").squeeze()
    X_test = pd.read_csv("./data/X_test.csv")
    y_test = pd.read_csv("./data/y_test.csv").squeeze()

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