import logging

from typing import Tuple
from typing_extensions import Annotated
import yaml

import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.base import ClassifierMixin
from sklearn.svm import SVC

from zenml import step

with open("steps/svc.yaml", "r") as f:
    model_config = yaml.safe_load(f)

@step(name="Load Iris")
def load_data() -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"]
]:
    """Loads the iris data from sklearn API.

    Returns:
        Tuple on annotated pandas dataframe splitted into X_train, X_test, y_train, and y_test.
    """
    try:
        logging.info("Loading iris daataset...")
        iris = load_iris(as_frame=True)
        print(iris)
        logging.info("Done loading, now splitting the data.")
        X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, 
                                                            test_size=0.2, shuffle=True, random_state=42)
        logging.info("Done splitting.")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error("Error while loading iris data: {}".format(e))
        raise e


@step(name="Train SVC")
def train_scv(
    X_train: pd.DataFrame,
    y_train: pd.Series) -> Tuple[
        Annotated[ClassifierMixin, "SVC Classifier"],
        Annotated[float, "Accuracy Score"]
    ]:
    """Trains SVC classifier.

    Args:
        X_train (pd.DataFrame): Pandas dataframe input features.
        y_train (pd.Series): Pandas series input target variable.

    Returns:
        ClassifierMixin: An isntance of base sklearn classifier.

    Raises: 
        Exception.
    """
    try:
        for params in model_config["parameters"]:
            logging.info("Training SVC classifier model.")            
            model = SVC(params)
            trained_svc = model.fit(X_train.to_numpy(), y_train.to_numpy())
            logging.info("Done training SVC.")
            logging.info("Calculatinfg training accuracy score...")
            acc_score = train_scv.score(X_train.to_numpy(), y_train.to_numpy())
            logging.info("Accuracy score: {.4f}".format(acc_score))
        return trained_svc, acc_score
    except Exception as e:
        logging.exception("Unepected error while training SVC: {}".format(e))