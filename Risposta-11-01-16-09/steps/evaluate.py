import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
import mlflow
import mlflow.sklearn

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    return f1, precision, recall

def evaluate_step():
    X_test = pd.read_csv('./data/X_test_transformed.csv')
    y_test = pd.read_csv('./data/y_test.csv')

    model = mlflow.sklearn.load_model("model")

    f1, precision, recall = evaluate_model(model, X_test, y_test)

    with mlflow.start_run():
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision_score", precision)
        mlflow.log_metric("recall_score", recall)
        print("Valutazione completata!")

if __name__ == "__main__":
    evaluate_step()
