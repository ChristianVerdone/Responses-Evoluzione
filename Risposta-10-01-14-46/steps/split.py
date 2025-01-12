import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

# Funzione per la divisione dei dati
def split_data(data, split_ratios):
    X = data.drop(columns=['is_red'])
    y = data['is_red']

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=1 - split_ratios[0], random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp,
                                                   test_size=split_ratios[2] / (split_ratios[1] + split_ratios[2]),
                                                   random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

# Carica i dati ingestiti
data = pd.read_csv('data_ingested.csv')

# Carica i parametri dal file local.yaml
with open('local.yaml', 'r') as file:
    params = yaml.safe_load(file)

split_ratios = params['SPLIT_RATIOS']

X_train, X_val, X_test, y_train, y_val, y_test = split_data(data, split_ratios)

# Salva i dati suddivisi
X_train.to_csv('X_train.csv', index=False)
X_val.to_csv('X_val.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_val.to_csv('y_val.csv', index=False)
y_test.to_csv('y_test.csv', index=False)
