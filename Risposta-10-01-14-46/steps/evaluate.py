import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics import f1_score, precision_score, recall_score

# Funzione per valutare il modello
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    return f1, precision, recall

# Carica i dati trasformati
X_test = pd.read_csv('X_test_transformed.csv')
y_test = pd.read_csv('y_test.csv')

# Carica il modello addestrato
model = mlflow.sklearn.load_model("model")

# Valutazione del modello
f1, precision, recall = evaluate_model(model, X_test, y_test)

# Avvia un run di MLflow
with mlflow.start_run():
    # Log dei parametri e delle metriche
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision_score", precision)
    mlflow.log_metric("recall_score", recall)

    # Output delle metriche
    print("F1 Score:", f1)
    print("Precision Score:", precision)
    print("Recall Score:", recall)

print("Esperimento completato!")
