import pandas as pd
from sklearn.model_selection import train_test_split
import logging

# Configura il logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def split_data(df, split_ratios=[0.8, 0.1, 0.1]):
    """
    Dividi il dataset in training, validation e test set.

    Parametri:
    - df (DataFrame): DataFrame contenente i dati da dividere.
    - split_ratios (list): Lista con le proporzioni per training, validation e test set.

    Restituisce:
    - train_df (DataFrame): DataFrame contenente il training set.
    - val_df (DataFrame): DataFrame contenente il validation set.
    - test_df (DataFrame): DataFrame contenente il test set.
    """
    train_ratio, val_ratio, test_ratio = split_ratios
    train_val_ratio = train_ratio + val_ratio

    train_val_df, test_df = train_test_split(df, test_size=test_ratio, random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=val_ratio / train_val_ratio, random_state=42)

    return train_df, val_df, test_df

if __name__ == "__main__":
    import yaml
    with open('local.yaml', 'r') as file:
        config = yaml.safe_load(file)
    split_config = {'split_ratios': config['SPLIT_RATIOS']}
    split_ratios = split_config['split_ratios']
    file_paths = config['INGEST_CONFIG']['location']
    data = pd.concat([pd.read_csv(file_path, delimiter=';') for file_path in file_paths], ignore_index=True)
    train_df, val_df, test_df = split_data(data, split_ratios)
    print(train_df.head())
    print(val_df.head())
    print(test_df.head())
