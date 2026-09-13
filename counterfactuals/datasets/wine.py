import pandas as pd

from counterfactuals.datasets.base import AbstractDataset


class WineDataset(AbstractDataset):
    def __init__(self, file_path: str = "data/wine.csv"):
        super().__init__()
        self._normal_init(file_path)

    def preprocess(self, raw_data: pd.DataFrame):
        """
        Preprocess the loaded data to X and y numpy arrays.
        """

        X = raw_data[raw_data.columns[1:]].to_numpy()
        y = raw_data[raw_data.columns[0]].to_numpy()

        self.feature_columns = list(raw_data.columns[1:])
        self.numerical_features = list(range(0, len(self.feature_columns)))
        self.numerical_columns = self.numerical_features
        self.categorical_features = []
        self.categorical_columns = []

        return X, y
