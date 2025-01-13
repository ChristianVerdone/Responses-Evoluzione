# register.py
import mlflow
import mlflow.sklearn

def register_model(model, allow_non_validated_model=False):
    if allow_non_validated_model:
        mlflow.sklearn.log_model(model, "model")
    else:
        # Logica per registrare solo modelli validati
        pass
