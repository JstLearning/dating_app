from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer


class Regressor(BaseEstimator):
    def __init__(self):
        self.model = RandomForestRegressor()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        y_pred = self.model.predict(X)
        return y_pred


class FeatureExtractor(BaseEstimator):
    def fit(self, X, y):
        self.nb_users_by_age = X.groupby("age").agg({"gender": "count"}).reset_index()
        self.nb_users_by_age.columns = ["age", "nb_users"]
        return self

    def transform(self, X):
        X = X.drop(columns=["locationCity"])
        return X.merge(self.nb_users_by_age, on="age", how="left")


def get_estimator():

    feature_extractor = FeatureExtractor()

    reg = Regressor()

    impute_missing_values = SimpleImputer(strategy="mean")

    pipe = make_pipeline(
        feature_extractor, impute_missing_values, StandardScaler(), reg
    )
    return pipe
