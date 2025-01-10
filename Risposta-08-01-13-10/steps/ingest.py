import pandas as pd


def ingest_data(file_paths):
    data_white = pd.read_csv(file_paths[0], delimiter=';')
    data_red = pd.read_csv(file_paths[1], delimiter=';')
    data_white["is_red"] = 0
    data_red["is_red"] = 1
    data = pd.concat([data_white, data_red], ignore_index=True)
    return data
