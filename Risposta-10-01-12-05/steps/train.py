from sklearn.ensemble import RandomForestClassifier
import logging

# Configurazione del logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model(X_train, y_train):
    """
    Addestra il modello utilizzando i dati forniti.

    Parametri:
    - X_train (DataFrame): Dati di input per l'addestramento.
    - y_train (Series): Colonna target per l'addestramento.

    Restituisce:
    - RandomForestClassifier: Modello addestrato.
    """
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

# Esempio di utilizzo
X_train = pd.read_csv("/data/X_train.csv")
y_train = pd.read_csv("/data/y_train.csv")
model = train_model(X_train, y_train)
