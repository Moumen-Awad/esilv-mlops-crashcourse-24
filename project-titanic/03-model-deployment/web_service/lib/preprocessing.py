from typing import List

import pandas as pd
from loguru import logger

from sklearn.feature_extraction import DictVectorizer

CATEGORICAL_COLS = ['Pclass', 'Sex', 'Embarked']

def encode_categorical_cols(df, encoders):
    """
    Encodes categorical columns in the DataFrame using pre-fitted encoders.
    Args:
        df (pd.DataFrame): The DataFrame containing the data with categorical columns to be encoded.
        encoders (dict): A dictionary containing pre-fitted encoders (e.g., LabelEncoder or DictVectorizer) for categorical columns.
                          Keys should be column names, and values should be the corresponding encoder objects.
    Returns:
        pd.DataFrame: The DataFrame with the categorical columns encoded.
    """
    for col in CATEGORICAL_COLS:
        if col in encoders:
            df[col] = encoders[col].transform(df[col])
        else:
            logger.warning(f"No encoder found for column {col}")
    return df