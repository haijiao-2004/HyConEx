import pandas as pd

from counterfactuals.datasets.base import AbstractDataset


class AuditDataset(AbstractDataset):
    def __init__(self, file_path: str = "data/audit.csv"):
        super().__init__()
        self._normal_init(file_path)

    def preprocess(self, raw_data: pd.DataFrame):
        """
        Preprocess the loaded data to X and y numpy arrays.
        """

        target_column = raw_data.columns[-1]
        self.feature_columns = list(raw_data.columns[2:-1])
        self.feature_columns.remove("Detection_Risk")
        self.numerical_columns = list(range(0, len(self.feature_columns)))
        self.categorical_columns = []

        row_per_class = sum(raw_data[target_column] == 1)
        raw_data = pd.concat(
            [
                raw_data[raw_data[target_column] == 0].sample(
                    row_per_class, random_state=42
                ),
                raw_data[raw_data[target_column] == 1],
            ]
        )
        X = raw_data[self.feature_columns].to_numpy()
        y = raw_data[target_column].to_numpy()
        return X, y
