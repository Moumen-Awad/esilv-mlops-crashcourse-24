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
    label_encoders = {}
    for col in CATEGORICAL_COLS:
        df[col] = df[col].fillna("Unknown")
        label_encoders[col] = LabelEncoder()
        df[col] = label_encoders[col].fit_transform(df[col])
    return df, label_encoders

@task
def extract_x_y(df, categorical_cols=None, dv=None, with_target=True):
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
def transform_categorical_cols(df, encoders): # for prediction
    for col, encoder in encoders.items():
        df[col] = df[col].fillna("Unknown")
        df[col] = encoder.transform(df[col])
    return df

@flow(name="Preprocess data")
def process_data(filepath: str, dv=None, with_target: bool = True, label_encoders=None) -> scipy.sparse.csr_matrix:
    """
    Load data from a parquet file
    Compute target (duration column) and apply threshold filters (optional)
    Turn features to sparce matrix
    :return The sparce matrix, the target' values and the
    dictvectorizer object if needed.
    """
    # DF
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