import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
import mlflow
import logging

# Configura il logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_model(model, X_test, y_test):
    """
    Valuta il modello.

    Parametri:
    - model: Modello addestrato.
    - X_test (DataFrame): DataFrame contenente le caratteristiche di test.
    - y_test (Series): Serie contenente il target di test.

    Restituisce:
    - dict: Dizionario contenente le metriche di valutazione.
    """
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    return {'f1_score': f1, 'precision_score': precision, 'recall_score': recall}

def register_model(model, metrics, thresholds, allow_non_validated_model=False):
    """
    Registra il modello se soddisfa le soglie.

    Parametri:
    - model: Modello addestrato.
    - metrics (dict): Metriche di valutazione del modello.
    - thresholds (dict): Soglie per ciascuna metrica.
    - allow_non_validated_model (bool): Se True, registra il modello anche se non soddisfa le soglie.
    """
    if all(metrics[metric] >= thresholds[metric] for metric in thresholds) or allow_non_validated_model:
        mlflow.sklearn.log_model(model, "model")
    else:
        logger.warning("Il modello non soddisfa le soglie e non verrà registrato.")

if __name__ == "__main__":
    import yaml
    with open('local.yaml', 'r') as file:
        config = yaml.safe_load(file)
    evaluate_config = {
        'thresholds': config['THRESHOLDS'],
        'allow_non_validated_model': config['ALLOW_NON_VALIDATED_MODEL']
    }
    thresholds = evaluate_config['thresholds']
    allow_non_validated_model = evaluate_config['allow_non_validated_model']
    file_paths = config['INGEST_CONFIG']['location']
    data = pd.concat([pd.read_csv(file_path, delimiter=';') for file_path in file_paths], ignore_index=True)
    target_col = config['target_col']
    X = data.drop(columns=[target_col])
    y = data[target_col]
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    register_model(model, metrics, thresholds, allow_non_validated_model)
    print(metrics)
