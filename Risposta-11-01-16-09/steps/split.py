import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(data, split_ratios=[0.8, 0.1, 0.1]):
    target_col = "is_red"
    X = data.drop(columns=[target_col])
    y = data[target_col]

    train_ratio, val_ratio, test_ratio = split_ratios
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=1 - train_ratio, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_ratio / (val_ratio + test_ratio), random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test

def split_step():
    data = pd.read_csv('./data/winequality.csv')
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(data, split_ratios=[0.8, 0.1, 0.1])

    X_train.to_csv('./data/X_train.csv', index=False)
    X_val.to_csv('./data/X_val.csv', index=False)
    X_test.to_csv('./data/X_test.csv', index=False)
    y_train.to_csv('./data/y_train.csv', index=False)
    y_val.to_csv('./data/y_val.csv', index=False)
    y_test.to_csv('./data/y_test.csv', index=False)
    print("Divisione completata!")

if __name__ == "__main__":
    split_step()
