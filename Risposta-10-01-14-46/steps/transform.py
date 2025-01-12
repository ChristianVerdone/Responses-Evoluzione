import pandas as pd

# Funzione per la trasformazione dei dati
def transform_data(data):
    # Esempio di trasformazioni personalizzate
    data = data.dropna()  # Rimuovi valori nulli
    return data

# Carica i dati suddivisi
X_train = pd.read_csv('X_train.csv')
X_val = pd.read_csv('X_val.csv')
X_test = pd.read_csv('X_test.csv')

# Trasforma i dati
X_train = transform_data(X_train)
X_val = transform_data(X_val)
X_test = transform_data(X_test)

# Salva i dati trasformati
X_train.to_csv('X_train_transformed.csv', index=False)
X_val.to_csv('X_val_transformed.csv', index=False)
X_test.to_csv('X_test_transformed.csv', index=False)
