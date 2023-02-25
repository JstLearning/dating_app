import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

problem_title = "Dating App"


###############
# Data reader #
###############


def _read_data(path):
    """Read the datasets and return it as a pandas DataFrame"""

    df2 = pd.read_csv(os.path.join(path, "data", "lovoo_v3_users_api-results.csv"))
    df3 = pd.read_csv(os.path.join(path, "data", "lovoo_v3_users_instances.csv"))

    df3 = df3[["connectedToFacebook", "userId"]]

    df = pd.merge(df2, df3, on="userId", how="left")
    df = df.drop_duplicates(subset="userId", keep=False)

    df.index = df["userId"]
    # Drop unuseful columns
    df = df.drop(
        [
            "userId",
            "name",
            "counts_g",
            "city",
            "location",
            "distance",
            "crypt",
            "freetext",
            "pictureId",
            "isSystemProfile",
        ],
        axis=1,
    )

    X = df.drop(["counts_kisses", "counts_profileVisits"], axis=1)
    y = df[["counts_kisses"]]
    return X, y


def get_data(path, test_size=0.2, random_state=42):
    """Return the data as a pandas DataFrame"""
    X, y = _read_data(path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train["counts_kisses"], y_test["counts_kisses"]


def get_train_data(path="."):
    """Return the training data as a pandas DataFrame"""
    X_train, _, y_train, _ = get_data(path)
    X_train = preprocess_data(X_train)
    return X_train, y_train


def get_test_data(path="."):
    """Return the test data as a pandas DataFrame"""
    _, X_test, _, y_test = get_data(path)
    X_test = preprocess_data(X_test)
    return X_test, y_test


######################
# Data Preprocessing #
######################


def transform_true_false_to_binary(df, column):
    """Transform a column with True/False values to a binary column"""

    df[column] = df[column].map({True: 1, False: 0})
    return df


def transform_gender_to_binary(df, column):
    """Transform the gender column to a binary column"""

    df[column] = df[column].map({"F": 0, "M": 1, "both": 2, "none": 3})
    df[column] = df[column].astype(int)
    return df


def transform_countries(df):
    df["country"] = df["country"].astype("category")
    df["country"] = df["country"].cat.codes
    return df


def transform_dates(df):
    df["lastOnlineDate"] = df["lastOnlineDate"].str[:10]
    df["lastOnlineDate"] = pd.to_datetime(df["lastOnlineDate"])
    df["lastOnlineDate"] = df["lastOnlineDate"].astype("int64")
    return df


def transform_description(df):
    """Convert the whazzup column into 0 if empty, 1 else"""
    df["description"] = df["whazzup"].fillna(0).astype(bool).astype(int)
    df = df.drop(["whazzup"], axis=1)
    return df


def preprocess_data(df):
    df_copy = df.copy()
    colums_binary = [
        "connectedToFacebook",
        "flirtInterests_chat",
        "flirtInterests_date",
        "flirtInterests_friends",
        "lang_es",
        "lang_fr",
        "lang_it",
        "lang_pt",
        "lang_de",
        "lang_en",
    ]
    for column in colums_binary:
        df_copy = transform_true_false_to_binary(df_copy, column)

    colums_gender = ["gender", "genderLooking"]
    for column in colums_gender:
        df_copy = transform_gender_to_binary(df_copy, column)

    df_copy = transform_countries(df_copy)
    df_copy = transform_dates(df_copy)
    df_copy = transform_description(df_copy)
    return df_copy


###########################
# Cross validation scheme #
###########################


def get_cv(X, y):
    """Return a cross validation scheme"""

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    return cv.split(X, y)


###############
# Score types #
###############
import rampwf as rw

score_types = [
    rw.score_types.RMSE(name="rmse"),
    rw.score_types.RelativeRMSE(name="relative_rmse"),
    rw.score_types.MARE(name="mare"),
]


############
# Workflow #
############
from rampwf.workflows.sklearn_pipeline import SKLearnPipeline
from rampwf.workflows.sklearn_pipeline import Estimator
import numpy as np

workflow = Estimator()


####################
# Predictions Type #
####################

Predictions = rw.prediction_types.make_regression()
