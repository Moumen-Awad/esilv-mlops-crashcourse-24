from typing import List
import pandas as pd
from sklearn.feature_extraction import DictVectorizer

C_C = ["PULocationID","DOLocationID","passenger_count"]

def encode_c_c(df:pd.DataFrame, Ca_Co: List[str] = None) -> pd.DataFrame:
    if Ca_Co is None:
        Ca_Co = C_C
    df[Ca_Co] = df[Ca_Co].fillna(-1).astype("int")
    df[Ca_Co] = df[Ca_Co].astype("str")
    return df

def exract_x_y(
        df: pd.DataFrame,
        Ca_Co: List[str] = None,
        dv: DictVectorizer = None,
        with_target: bool = True,
) -> dict:
    if Ca_Co is None:
        Ca_Co = C_C
    dicts = df[Ca_Co].to_dict(orient="records")

    y = None
    if with_target:
        if dv is None:
            dv = DictVectorizer()
            dv.fit(dicts)
        y = df["duration"].values
    x = dv.transform(dicts)
    return x,y,dv