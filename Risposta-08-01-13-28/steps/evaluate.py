from sklearn.metrics import f1_score, precision_score, recall_score


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    return f1, precision, recall


f1, precision, recall = evaluate_model(model, X_test, y_test)
print("F1 Score:", f1)
print("Precision Score:", precision)
print("Recall Score:", recall)
