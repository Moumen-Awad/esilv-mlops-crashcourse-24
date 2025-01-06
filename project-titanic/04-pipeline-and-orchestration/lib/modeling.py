# Import important libraries
import numpy as np
import scipy.sparse
from prefect import task
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


@task(name="Train model")
def train_model(X: scipy.sparse.csr_matrix, y: np.ndarray) -> RandomForestClassifier:
    """
    Trains a RandomForestClassifier model using the provided feature matrix and target labels.
    Parameters:
        X (scipy.sparse.csr_matrix): A sparse matrix containing the feature set for training.
        y (np.ndarray): A 1-dimensional array containing the target labels.
    Returns:
        RandomForestClassifier: The trained model.
    """
    model = RandomForestClassifier()
    model.fit(X, y)
    return model


@task(name="Make predictions")
def predict(X: scipy.sparse.csr_matrix, model: RandomForestClassifier) -> np.ndarray:
    """
    Makes predictions using a trained RandomForestClassifier model.
    Parameters:
        X (scipy.sparse.csr_matrix): A sparse matrix containing the feature set for prediction.
        model (RandomForestClassifier): The trained RandomForestClassifier model.
    Returns:
        np.ndarray: An array of predicted labels.
    """
    return model.predict(X)


@task(name="Evaluate model")
def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the accuracy score between the true labels and predicted labels.
    Parameters:
        y_true (np.ndarray): A 1-dimensional array containing the true labels.
        y_pred (np.ndarray): A 1-dimensional array containing the predicted labels.
    Returns:
        float: The accuracy score as a percentage.
    """    
    return accuracy_score(y_true, y_pred)
