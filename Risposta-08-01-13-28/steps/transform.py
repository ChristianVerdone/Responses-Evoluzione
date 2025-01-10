def transform_data(data):
    data = data.dropna()
    return data


X_train = transform_data(X_train)
X_val = transform_data(X_val)
X_test = transform_data(X_test)
