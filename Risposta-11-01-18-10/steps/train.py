import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import logging

# Configura il logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model(X_train, y_train, estimator_params=None):
    """
    Addestra il modello.

    Parametri:
    - X_train (DataFrame): DataFrame contenente le caratteristiche di training.
    - y_train (Series): Serie contenente il target di training.
    - estimator_params (dict): Parametri dello stimatore.

    Restituisce:
    - model: Modello addestrato.
    """
    if estimator_params is None:
        estimator_params = {}

    model = RandomForestClassifier(**estimator_params)
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    import yaml
    with open('local.yaml', 'r') as file:
        config = yaml.safe_load(file)
    train_config = {'estimator_params': config['ESTIMATOR_PARAMS']}
    estimator_params = train_config['estimator_params']
    file_paths = config['INGEST_CONFIG']['location']
    data = pd.concat([pd.read_csv(file_path, delimiter=';') for file_path in file_paths], ignore_index=True)
    target_col = config['target_col']
    X = data.drop(columns=[target_col])
    y = data[target_col]
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train, estimator_params)
    print(model)
