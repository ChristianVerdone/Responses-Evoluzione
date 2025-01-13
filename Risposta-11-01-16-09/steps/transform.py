import pandas as pd

def transform_data(data):
    # Esempio di trasformazioni personalizzate
    data = data.dropna()  # Rimuovi valori nulli
    return data

def transform_step():
    X_train = pd.read_csv('./data/X_train.csv')
    X_val = pd.read_csv('./data/X_val.csv')
    X_test = pd.read_csv('./data/X_test.csv')

    X_train = transform_data(X_train)
    X_val = transform_data(X_val)
    X_test = transform_data(X_test)

    X_train.to_csv('./data/X_train_transformed.csv', index=False)
    X_val.to_csv('./data/X_val_transformed.csv', index=False)
    X_test.to_csv('./data/X_test_transformed.csv', index=False)
    print("Trasformazione completata!")

if __name__ == "__main__":
    transform_step()
