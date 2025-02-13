from sklearn.metrics import f1_score, precision_score, recall_score
import mlflow
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_model(model, X_test, y_test, thresholds, allow_non_validated):
    """Valuta il modello e gestisci la registrazione."""
    try:
        logger.info("Valutazione modello...")
        y_pred = model.predict(X_test)

        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        # Log delle metriche in MLflow
        mlflow.log_metrics({
            "f1_score": f1,
            "precision_score": precision,
            "recall_score": recall
        })

        # Controllo delle soglie
        is_valid = (
                f1 >= thresholds["f1_score"] and
                precision >= thresholds["precision_score"] and
                recall >= thresholds["recall_score"]
        )

        if not is_valid and not allow_non_validated:
            logger.warning("Modello non valido: metriche sotto le soglie.")

        return is_valid
    except Exception as e:
        logger.error(f"Errore durante la valutazione: {e}")
        raise