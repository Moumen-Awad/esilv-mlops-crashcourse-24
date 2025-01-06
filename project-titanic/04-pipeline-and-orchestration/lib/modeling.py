import numpy as np
import scipy.sparse
from prefect import task
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


@task(name="Train model")
def train_model(X: scipy.sparse.csr_matrix, y: np.ndarray) -> RandomForestClassifier:
    """Train and return a RandomForestClassifier model"""
    lr = RandomForestClassifier()
    lr.fit(X, y)
    return lr


@task(name="Make predictions")
def predict(X: scipy.sparse.csr_matrix, model: RandomForestClassifier) -> np.ndarray:
    """Make predictions with a trained model"""
    return model.predict(X)


@task(name="Evaluate model")
def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate mean squared error for two arrays"""
    return accuracy_score(y_true, y_pred, squared=False)
