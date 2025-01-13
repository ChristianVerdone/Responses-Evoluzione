import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn

# Funzione per l'addestramento del modello
def train_model(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

# Carica i dati trasformati
X_train = pd.read_csv('X_train_transformed.csv')
y_train = pd.read_csv('y_train.csv')

# Addestramento del modello
model = train_model(X_train, y_train)

# Avvia un run di MLflow
with mlflow.start_run():
    # Log del modello con MLflow
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_param("model_type", "RandomForest")
