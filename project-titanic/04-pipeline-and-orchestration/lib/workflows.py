# Import required libraries
import os
from typing import Optional
import numpy as np
from helpers import load_pickle, save_pickle
from loguru import logger
from modeling import evaluate_model, predict, train_model
from prefect import flow
from preprocessing import process_data
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier

@flow(name="Train model")
def train_model_workflow(
    data_filepath: str,
    artifacts_filepath: Optional[str] = None,
) -> dict:
    """
    A workflow to process data, train a model, make predictions, evaluate the model, 
    and optionally save artifacts to a specified location.
    Parameters:
        data_filepath (str): Path to the CSV file containing the training and test data.
        artifacts_filepath (Optional[str]): Path to save the artifacts (model, DictVectorizer, and label encoders). 
                                            If None, artifacts will not be saved.
    Returns:
        dict: A dictionary containing the trained model, DictVectorizer, label encoders, and accuracy score.
    """
    logger.info("Processing training data...")
    X_train, y_train, dv, label_encoders = process_data(filepath=data_filepath, with_target=True)
    logger.info("Processing test data...")
    X_test, y_test, _, _ = process_data(filepath=data_filepath, with_target=True, dv=dv, label_encoders=label_encoders)
    logger.info("Training model...")
    model = train_model(X_train, y_train)
    logger.info("Making predictions and evaluating...")
    y_pred = predict(X_test, model) 
    accuarcy = evaluate_model(y_test, y_pred)

    if artifacts_filepath is not None:
        logger.info(f"Saving artifacts to {artifacts_filepath}...")
        save_pickle(os.path.join(artifacts_filepath, "dv.pkl"), dv)
        save_pickle(os.path.join(artifacts_filepath, "encoders.pkl"), label_encoders)
        save_pickle(os.path.join(artifacts_filepath, "model.pkl"), model)

    return {"model": model, "dv": dv, "label_encoders": label_encoders, "accuarcy": accuarcy}


@flow(name="Batch predict", retries=1, retry_delay_seconds=30)
def batch_predict_workflow(
    input_filepath: str,
    model: Optional[RandomForestClassifier] = None,
    dv: Optional[DictVectorizer] = None,
    label_encoders: Optional[dict] = None,
    artifacts_filepath: Optional[str] = None,
) -> np.ndarray:
    """
    Performs batch prediction on a new dataset using a pre-trained model and saved artifacts.
    This function loads the necessary pre-trained model, `DictVectorizer`, and `LabelEncoder` from the specified file paths if they are not provided. It then processes the input data, applies feature transformation, and makes predictions using the loaded model.
    Args:
        input_filepath (str): The path to the input dataset (CSV file) for prediction.
        model (RandomForestClassifier, optional): The pre-trained machine learning model. If not provided, the model is loaded from the artifacts folder.
        dv (DictVectorizer, optional): The pre-fitted `DictVectorizer` for transforming categorical features. If not provided, it is loaded from the artifacts folder.
        label_encoders (dict, optional): A dictionary of pre-fitted `LabelEncoder` objects used for transforming categorical features. If not provided, they are loaded from the artifacts folder.
        artifacts_filepath (str, optional): The directory where the model, `DictVectorizer`, and encoders are saved as pickle files.
    Returns:
        np.ndarray: The predicted values based on the input dataset.
    """
    if dv is None:
        dv = load_pickle(os.path.join(artifacts_filepath, "dv.pkl"))
    if model is None:
        model = load_pickle(os.path.join(artifacts_filepath, "model.pkl"))
    if label_encoders is None:
        label_encoders = load_pickle(os.path.join(artifacts_filepath, "encoders.pkl"))

    X, _, _, _ = process_data(filepath=input_filepath, with_target=False, dv=dv, label_encoders=label_encoders)
    y_pred = predict(X, model)

    return y_pred


if __name__ == "__main__":
    from config import DATA_DIRPATH, MODELS_DIRPATH

    train_model_workflow(
        data_filepath=os.path.join(DATA_DIRPATH, "train_and_test2.csv"),
        artifacts_filepath=MODELS_DIRPATH,
    )

    batch_predict_workflow(
        input_filepath=os.path.join(DATA_DIRPATH, "train_and_test2.csv"),
        artifacts_filepath=MODELS_DIRPATH,
    )
