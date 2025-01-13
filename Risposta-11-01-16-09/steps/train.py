import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn

def train_model(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

def train_step():
    X_train = pd.read_csv('./data/X_train_transformed.csv')
    y_train = pd.read_csv('./data/y_train.csv')

    with mlflow.start_run():
        model = train_model(X_train, y_train)
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_param("model_type", "RandomForest")
        print("Addestramento completato!")

if __name__ == "__main__":
    train_step()
