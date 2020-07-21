import gc

import numpy as np
import pandas as pd


def time_features(df, date_col):
    attrs = [
        # "year",
        "quarter",
        # "month",
        "week",
        "day",
        "dayofweek",
    ]

    for attr in attrs:
        df[attr] = getattr(df[date_col].dt, attr).astype(np.int64)

    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(np.int8)
    return df


def downcast_numeric_cols(df):
    int_cols = df.select_dtypes(include=["int"]).columns
    float_cols = df.select_dtypes(include=["float"]).columns

    for col in int_cols:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    for col in float_cols:
        df[col] = pd.to_numeric(df[col], downcast="float")

    gc.collect()

    return df


def label_encoder(df, cols):
    for col in cols:
        print("Processing column: {}".format(col))
        df[col] = df[col].astype("category")

    return df


def convert_to_date(df, cols):
    for col in cols:
        df[col] = pd.to_datetime(df[col])

    return df
