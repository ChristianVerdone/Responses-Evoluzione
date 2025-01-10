import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(df, split_ratios):
    """
    Divide il DataFrame in training, validation e test set.

    Args:
        df (pd.DataFrame): DataFrame da dividere.
        split_ratios (list): Lista di tre numeri che rappresentano le proporzioni per training, validation e test.

    Returns:
        tuple: Una tupla contenente i DataFrame di training, validation e test.
    """
    train_ratio, val_ratio, test_ratio = split_ratios

    if sum(split_ratios) != 1.0:
        raise ValueError("Le proporzioni di split devono sommare a 1.")

    train_df, temp_df = train_test_split(df, train_size=train_ratio, random_state=42)
    val_df, test_df = train_test_split(temp_df, train_size=val_ratio / (val_ratio + test_ratio), random_state=42)

    return train_df, val_df, test_df
