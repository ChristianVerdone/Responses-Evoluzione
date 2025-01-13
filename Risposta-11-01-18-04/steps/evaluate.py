from sklearn.metrics import f1_score, precision_score, recall_score
import logging
import mlflow
import mlflow.sklearn

# Configura il logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    logger.info(f"F1 Score: {f1}, Precision: {precision}, Recall: {recall}")
    return f1, precision, recall

def register_model(model, validation_passed, allow_non_validated_model=False):
    if validation_passed or allow_non_validated_model:
        mlflow.sklearn.log_model(model, "model")
        logger.info("Modello registrato con successo.")
    else:
        logger.warning("Modello non registrato perché non ha superato la validazione.")

# Esempio di utilizzo
f1, precision, recall = evaluate_model(model, X_test, y_test)
thresholds = {'f1_score': 0.7, 'precision_score': 0.7, 'recall_score': 0.7}
validation_passed = (f1 >= thresholds['f1_score']) and (precision >= thresholds['precision_score']) and (recall >= thresholds['recall_score'])
allow_non_validated_model = False
register_model(model, validation_passed, allow_non_validated_model)
