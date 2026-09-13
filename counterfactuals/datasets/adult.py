import pandas as pd

from counterfactuals.datasets.base import AbstractDataset


class AdultDataset(AbstractDataset):
    def __init__(self, file_path: str = "data/adult.csv"):
        super().__init__()
        self._normal_init(file_path)

    def preprocess(self, raw_data: pd.DataFrame):
        """
        Preprocess the loaded data to X and y numpy arrays.
        """
        self.feature_columns = [
            # Continuous
            "age",
            "hours_per_week",
            # Categorical
            "workclass",
            "education",
            "marital_status",
            "occupation",
            "race",
            "gender",
        ]
        self.numerical_columns = list(range(0, 2))
        self.categorical_columns = list(range(2, len(self.feature_columns)))
        target_column = "income"

        # Downsample to minor class
        # raw_data = raw_data.dropna(subset=self.feature_columns)
        # row_per_class = sum(raw_data[target_column] == 1)
        # raw_data = pd.concat(
        #     [
        #         raw_data[raw_data[target_column] == 0].sample(
        #             row_per_class, random_state=42
        #         ),
        #         raw_data[raw_data[target_column] == 1],
        #     ]
        # )

        X = raw_data[self.feature_columns].to_numpy()
        y = raw_data[target_column].to_numpy()

        return X, y
