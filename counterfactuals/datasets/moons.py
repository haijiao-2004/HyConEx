import pandas as pd

from counterfactuals.datasets.base import AbstractDataset


class MoonsDataset(AbstractDataset):
    def __init__(self, file_path: str = "data/moons.csv"):
        super().__init__()
        self._normal_init(file_path)

    def preprocess(self, raw_data: pd.DataFrame):
        """
        Preprocess the loaded data to X and y numpy arrays.
        """
        self.categorical_columns = []
        self.numerical_columns = [0, 1]
        X = raw_data[raw_data.columns[:-1]].to_numpy()
        y = raw_data[raw_data.columns[-1]].to_numpy()

        self.numerical_features = [0, 1]
        self.categorical_features = []
        self.actionable_features = [0, 1]
        return X, y
