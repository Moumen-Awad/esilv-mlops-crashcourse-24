from typing import List, Tuple

import numpy as np
import pandas as pd
import scipy.sparse
from config import CATEGORICAL_COLS
from loguru import logger
from sklearn.preprocessing import LabelEncoder
from prefect import flow, task
from sklearn.feature_extraction import DictVectorizer


@task
def encode_categorical_cols(df):
    """
    Encodes categorical columns in the DataFrame using LabelEncoder.
    This function fills missing values in categorical columns with "Unknown" and then encodes the categorical values into numeric labels using LabelEncoder.
    Args:
        df (pd.DataFrame): The DataFrame containing categorical columns to be encoded. The columns to encode are specified by `CATEGORICAL_COLS`.
    Returns:
        pd.DataFrame: The DataFrame with the categorical columns encoded as numeric labels.
        dict: A dictionary containing the LabelEncoders used for each categorical column. The keys are the column names, and the values are the fitted LabelEncoder objects.
    """
    label_encoders = {}
    for col in CATEGORICAL_COLS:
        df[col] = df[col].fillna("Unknown")
        label_encoders[col] = LabelEncoder()
        df[col] = label_encoders[col].fit_transform(df[col])
    return df, label_encoders

@task
def extract_x_y(df, categorical_cols=None, dv=None, with_target=True):
    """
    Extracts features (X) and target (y) from the given DataFrame.
    This function prepares the data by extracting the categorical features from the DataFrame and transforming them using a `DictVectorizer`. If `with_target` is set to True, the target variable is also extracted.
    Args:
    df (pd.DataFrame): The DataFrame containing the data.
        categorical_cols (list, optional): A list of categorical column names to be used for feature extraction. Defaults to `CATEGORICAL_COLS` if not provided.
        dv (DictVectorizer, optional): A pre-fitted `DictVectorizer` to be used for transforming the categorical columns. If not provided, a new `DictVectorizer` will be created and fitted.
        with_target (bool, optional): Whether to include the target variable in the return value. Defaults to True.
    Returns:
        tuple:
            - x (scipy.sparse.csr_matrix): The transformed feature matrix (X) after applying `DictVectorizer` to the categorical columns.
            - y (np.ndarray or None): The target variable (y) if `with_target` is True, otherwise None.
            - dv (DictVectorizer): The fitted `DictVectorizer` used to transform the categorical columns.
    """
    if categorical_cols is None:
        categorical_cols = CATEGORICAL_COLS
    
    dicts = df[categorical_cols].to_dict(orient="records")
    y = None

    if with_target:
        if dv is None:
            dv = DictVectorizer()
            dv.fit(dicts)
        y = df["2urvived"].values

    x = dv.transform(dicts)
    return x, y, dv

@task
def transform_categorical_cols(df, encoders):
    """
    Transforms categorical columns in the DataFrame using pre-fitted encoders.
    This function applies the pre-fitted encoders (LabelEncoders) to transform categorical columns in the DataFrame. It also handles missing values by filling them with the label "Unknown" before applying the transformation.
    Args:
        df (pd.DataFrame): The DataFrame containing the categorical columns to be transformed.
        encoders (dict): A dictionary where the keys are column names and the values are the pre-fitted `LabelEncoder` objects used for transforming the categorical columns.
    Returns:
        pd.DataFrame: The DataFrame with transformed categorical columns.
    """
    for col, encoder in encoders.items():
        df[col] = df[col].fillna("Unknown")
        df[col] = encoder.transform(df[col])
    return df

@flow(name="Preprocess data")
def process_data(filepath: str, dv=None, with_target: bool = True, label_encoders=None) -> scipy.sparse.csr_matrix:
    """
    Processes data for training or prediction by handling categorical encoding and feature extraction.
    This function reads the dataset, selects relevant columns, handles missing values, and performs encoding for categorical features. It returns processed features (X) and target variable (y) for training or just the features for prediction. The function also handles saving and returning the `DictVectorizer` and `LabelEncoder` objects for later use.

    Args:
        filepath (str): The path to the CSV file containing the dataset.
        dv (DictVectorizer, optional): A pre-fitted `DictVectorizer` to transform categorical columns. If not provided, a new one will be created.
        with_target (bool, optional): Whether to extract the target variable `y`. Defaults to True, meaning the function will be used for training.
        label_encoders (dict, optional): A dictionary of pre-fitted `LabelEncoder` objects used for encoding categorical columns. Used only for prediction.
    Returns:
        tuple:
            - x (scipy.sparse.csr_matrix): The transformed feature matrix (X) after applying encoding and vectorization.
            - y (np.ndarray or None): The target variable `y` if `with_target` is True, otherwise None (for prediction).
            - dv (DictVectorizer): The fitted `DictVectorizer` used to transform categorical features.
            - label_encoders (dict): The dictionary of `LabelEncoder` objects used for encoding categorical columns.
    """
    df = pd.read_csv(filepath)
    selected_cols = ['Age', 'Fare', 'Sex', 'sibsp', 'Parch', 'Pclass', 'Embarked', '2urvived']
    df = df[selected_cols].dropna()
    if with_target: #train-test
        logger.debug(f"{filepath} | Encoding categorical columns...")
        df1, label_encoders = encode_categorical_cols(df)
        logger.debug(f"{filepath} | Extracting X and y...")
        x, y, dv = extract_x_y(df1, dv=dv)
        return x, y, dv, label_encoders
    else: #predict
        logger.debug(f"{filepath} | Encoding categorical columns...")
        df1 = transform_categorical_cols(df, label_encoders)
        logger.debug(f"{filepath} | Extracting X and y...")
        x, _, dv = extract_x_y(df1, dv=dv, with_target=with_target)
        return x, None, dv, label_encoders  # Return None for y in prediction