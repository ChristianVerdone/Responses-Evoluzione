import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import logging

# Configura il logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def train_model(X_train, y_train):
    """
    Addestra un modello RandomForest sui dati di training.

    :param X_train: DataFrame delle caratteristiche di training
    :param y_train: Serie dei target di training
    :return: Modello addestrato
    """
    logger.info("Addestramento del modello RandomForest.")
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

'''
# Esempio di utilizzo
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
model = train_model(X_train, y_train)'''