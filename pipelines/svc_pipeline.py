from zenml import pipeline

from steps.svc_steps import load_data, train_scv

@pipeline(name="Simple SVC", enable_cache=True)
def run_svc():
    X_train, X_test, y_train, y_test = load_data()
    trained_svc, accuracy = train_scv(X_train, X_test)