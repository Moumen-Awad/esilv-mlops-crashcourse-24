"""
This module contains utility functions for loading various objects from disk, such as machine learning models, preprocessors, and encoders. 

Each function uses the `lru_cache` decorator to cache the loaded objects, ensuring that the same object is not reloaded multiple times, which improves efficiency when the objects are accessed repeatedly during the execution.

Functions:
- `load_preprocessor(filepath)`: Loads and caches a preprocessor object (e.g., DictVectorizer) from the specified file.
- `load_model(filepath)`: Loads and caches a trained model (e.g., RandomForestClassifier) from the specified file.
- `load_encoder(filepath)`: Loads and caches an encoder (e.g., LabelEncoder) from the specified file.

All functions use pickle to deserialize the objects from their respective files. The file paths are passed as arguments to each function. Logging is used to provide information about which file is being loaded.
"""

import os
import pickle
from functools import lru_cache
from loguru import logger


@lru_cache
def load_preprocessor(filepath: os.PathLike):
    logger.info(f"Loading preprocessor from {filepath}")
    with open(filepath, "rb") as f:
        return pickle.load(f)

@lru_cache
def load_model(filepath: os.PathLike):
    logger.info(f"Loading model from {filepath}")
    with open(filepath, "rb") as f:
        return pickle.load(f)
    
@lru_cache
def load_encoder(filepath: os.PathLike):
    logger.info(f"Loading encoder from {filepath}")
    with open(filepath, "rb") as f:
        return pickle.load(f)