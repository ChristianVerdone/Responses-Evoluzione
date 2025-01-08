import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

# Configura il logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def train_model(X_train, y_train):
    """
    Addestra il modello RandomForestClassifier.

    :param X_train: DataFrame contenente le caratteristiche di training
    :param y_train: Series contenente il target di training
    :return: Modello addestrato
    """
    logger.info("Addestramento del modello RandomForestClassifier")
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Valuta il modello utilizzando f1_score, precision_score e recall_score.

    :param model: Modello addestrato
    :param X_test: DataFrame contenente le caratteristiche di test
    :param y_test: Series contenente il target di test
    :return: Tuple contenente f1, precision e recall
    """
    logger.info("Valutazione del modello")
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    return f1, precision, recall


# Esempio di utilizzo delle funzioni
if __name__ == "__main__":
    file_paths = ["./data/winequality-white.csv", "./data/winequality-red.csv"]
    data = pd.read_csv(file_paths[0], delimiter=';')
    target_col = "is_red"
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(data, target_col)

    model = train_model(X_train, y_train)
    f1, precision, recall = evaluate_model(model, X_test, y_test)

    print("F1 Score:", f1)
    print("Precision Score:", precision)
    print("Recall Score:", recall)