from sklearn.ensemble import RandomForestClassifier

def custom_estimator(estimator_params=None):
    if estimator_params is None:
        estimator_params = {}
    
    return RandomForestClassifier(random_state=42, **estimator_params)