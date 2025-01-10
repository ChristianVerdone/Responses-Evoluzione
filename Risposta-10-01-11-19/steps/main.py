import mlflow
import mlflow.sklearn
from ingest import ingest_data
from split import split_data
from transform import transform_data
from train import train_model
from evaluate import evaluate_model
import yaml

# Carica la configurazione dal file YAML
with open("local.yaml", "r") as f:
    config = yaml.safe_load(f)

# Step di ingestione
data = ingest_data(config["INGEST_CONFIG"]["location"])

# Separazione in caratteristiche e target
X = data.drop(columns=[config["target_col"]])
y = data[config["target_col"]]

# Step di split
X_train, X_val, X_test, y_train, y_val, y_test = split_data(
    X, y, config["SPLIT_CONFIG"]["split_ratios"]
)

# Step di trasformazione
X_train = transform_data(X_train)
X_val = transform_data(X_val)
X_test = transform_data(X_test)

# Avvio del run di MLflow
with mlflow.start_run():
    # Step di training
    model = train_model(X_train, y_train, config["TRAIN_PARAMS"]["model_params"])
    mlflow.sklearn.log_model(model, "model")

    # Step di valutazione
    metrics = evaluate_model(model, X_test, y_test)
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, metric_value)

    # Log dei parametri
    mlflow.log_param("model_type", config["TRAIN_PARAMS"]["model_type"])
