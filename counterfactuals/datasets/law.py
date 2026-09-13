import pandas as pd

from counterfactuals.datasets.base import AbstractDataset


class LawDataset(AbstractDataset):
    def __init__(self, file_path: str = "data/law.csv"):
        super().__init__()
        self._normal_init(file_path)

    def preprocess(self, raw_data: pd.DataFrame):
        """
        Preprocess the loaded data to X and y numpy arrays.
        """
        # self.feature_columns = ["lsat", "gpa", "zfygpa", "sex", "race"]
        self.feature_columns = ["lsat", "gpa", "zfygpa"]
        target_column = "pass_bar"
        self.numerical_columns = list(range(0, 3))
        self.categorical_columns = list(range(3, len(self.feature_columns)))
        # self.categorical_columns = []

        # Downsample to minor class
        raw_data = raw_data.dropna(subset=self.feature_columns)
        row_per_class = sum(raw_data[target_column] == 0)
        raw_data = pd.concat(
            [
                raw_data[raw_data[target_column] == 0],
                raw_data[raw_data[target_column] == 1].sample(
                    row_per_class, random_state=42
                ),
            ]
        )

        X = raw_data[self.feature_columns].to_numpy()
        y = raw_data[target_column].to_numpy()
        return X, y
