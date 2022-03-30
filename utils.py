import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def one_hot_encode(df, columns, drop_columns):
    """
    One hot encode a dataframe with categorical columns
    """
    for col in columns:
        dummies = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df, dummies], axis=1)

    df.drop(drop_columns, axis=1, inplace=True)

    return df


def train_test_split_df(df, test_size=0.2):
    """
    Split dataframe into train and test data
    """
    df_train, df_test = df[:round(len(df) * (1 - test_size))], df[round(
        len(df) * (1 - test_size)):]
    print(f"len train df: {len(df_train)}")
    print(f"len test df: {len(df_test)}")

    return df_train, df_test


def standardize(df, columns):
    """
    Standardize dataframe with columns
    """
    scaler = StandardScaler()
    scaler.fit(df[columns])
    df[columns] = scaler.transform(df[columns])

    return df, scaler


def divide_train_target(df, data_columns, date_fields, target):
    """
    Divide dataframe into train and target
    convert train and target to numpy arrays
    """
    features_columns = data_columns
    columns_to_drop = date_fields
    # combine elements of date_fields and target
    columns_to_drop.extend(target)

    for col in columns_to_drop:
        features_columns.remove(col)

    x, y = df[features_columns].to_numpy(), df[target].to_numpy()

    return x, y


def train_validation_split(x, y, test_size=0.2, random_state=42):
    """
    Split dataframe into train and validation data
    """

    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=test_size, random_state=random_state)

    return x_train, x_valid, y_train, y_valid