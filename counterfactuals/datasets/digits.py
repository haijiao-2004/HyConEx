import pandas as pd
from sklearn.datasets import load_digits

from counterfactuals.datasets.base import AbstractDataset


class DigitsDataset(AbstractDataset):
    def __init__(self, file_path: str = "data/digits.csv"):
        super().__init__()
        self.raw_data = pd.DataFrame()
        self._normal_init(None)

    def preprocess(self, raw_data: pd.DataFrame):
        """
        Preprocess the loaded data to X and y numpy arrays.
        """
        X, y = load_digits(n_class=10, return_X_y=True)

        X = X.reshape(X.shape[0], -1)

        self.numerical_columns = list(range(0, X.shape[1]))
        self.categorical_columns = []

        return X, y
