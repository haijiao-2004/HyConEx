import numpy as np
import pandas as pd

from counterfactuals.datasets.base import AbstractDataset


class HelocDataset(AbstractDataset):
    def __init__(self, file_path: str = "data/heloc.csv"):
        super().__init__()
        self._normal_init(file_path)

    def preprocess(self, raw_data: pd.DataFrame):
        """
        Preprocess the loaded data to X and y numpy arrays.
        """
        raw_data = raw_data.replace(-9, np.nan).dropna()
        target_column = "RiskPerformance"
        self.feature_columns = raw_data.columns.drop(target_column)

        self.numerical_columns = list(range(0, len(self.feature_columns)))
        self.categorical_columns = []

        X = raw_data[self.feature_columns].to_numpy()
        y = raw_data[target_column].to_numpy()

        return X, y
