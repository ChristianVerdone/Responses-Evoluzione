from sklearn.metrics import f1_score, precision_score, recall_score
import logging

# Configurazione del logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_model(model, X_test, y_test):
    """
    Valuta il modello utilizzando i dati di test.

    Parametri:
    - model (RandomForestClassifier): Modello addestrato.
    - X_test (DataFrame): Dati di input per il test.
    - y_test (Series): Colonna target per il test.

    Restituisce:
    - tuple: Tupla contenente f1_score, precision_score e recall_score.
    """
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    return f1, precision, recall

# Esempio di utilizzo
X_test = pd.read_csv("/data/X_test.csv")
y_test = pd.read_csv("/data/y_test.csv")
model = RandomForestClassifier()
f1, precision, recall = evaluate_model(model, X_test, y_test)
