from sklearn.ensemble import RandomForestClassifier
import logging

# Configura il logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model(X_train, y_train, model_type="RandomForest"):
    if model_type == "RandomForest":
        model = RandomForestClassifier()
    else:
        raise ValueError(f"Model type '{model_type}' not supported.")

    model.fit(X_train, y_train)
    return model

# Esempio di utilizzo
model = train_model(X_train, y_train)
logger.info("Modello addestrato con successo.")
