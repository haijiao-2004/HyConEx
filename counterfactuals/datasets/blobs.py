import pandas as pd

from counterfactuals.datasets.base import AbstractDataset


class BlobsDataset(AbstractDataset):
    def __init__(self, file_path: str = "data/blobs.csv"):
        super().__init__()
        self._normal_init(file_path)

    def preprocess(self, raw_data: pd.DataFrame):
        """
        Preprocess the loaded data to X and y numpy arrays.
        """

        X = raw_data[raw_data.columns[:-1]].to_numpy()
        y = raw_data[raw_data.columns[-1]].to_numpy()

        self.numerical_features = [0, 1]
        self.numerical_columns = [0, 1]
        self.categorical_features = []
        self.actionable_features = [0, 1]
        self.categorical_columns = []

        return X, y
