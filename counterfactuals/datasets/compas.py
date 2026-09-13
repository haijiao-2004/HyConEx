import numpy as np
import pandas as pd

from counterfactuals.datasets.base import AbstractDataset


class CompasDataset(AbstractDataset):
    def __init__(self, file_path: str = "data/compas_two_years.csv"):
        super().__init__()
        self._normal_init(file_path)

    def preprocess(self, raw_data: pd.DataFrame):
        """
        Preprocess the loaded data to X and y numpy arrays.
        """

        self.feature_columns = [
            # Continuous
            "age",
            "priors_count",
            "days_b_screening_arrest",
            "length_of_stay",
            "is_recid",
            "is_violent_recid",
            "two_year_recid",
            # Categorical
            "c_charge_degree",
            "sex",
            "race",
        ]
        self.numerical_columns = list(range(0, 7))
        self.categorical_columns = list(range(7, len(self.feature_columns)))
        target_column = "class"

        raw_data["days_b_screening_arrest"] = np.abs(
            raw_data["days_b_screening_arrest"]
        )
        raw_data["c_jail_out"] = pd.to_datetime(raw_data["c_jail_out"])
        raw_data["c_jail_in"] = pd.to_datetime(raw_data["c_jail_in"])
        raw_data["length_of_stay"] = np.abs(
            (raw_data["c_jail_out"] - raw_data["c_jail_in"]).dt.days
        )
        raw_data["length_of_stay"].fillna(
            raw_data["length_of_stay"].value_counts().index[0], inplace=True
        )
        raw_data["days_b_screening_arrest"].fillna(
            raw_data["days_b_screening_arrest"].value_counts(
            ).index[0], inplace=True
        )
        raw_data["length_of_stay"] = raw_data["length_of_stay"].astype(int)
        raw_data["days_b_screening_arrest"] = raw_data[
            "days_b_screening_arrest"
        ].astype(int)
        # raw_data = raw_data[raw_data["score_text"] != "Medium"]
        # raw_data["class"] = pd.get_dummies(raw_data["score_text"])["High"].astype(int)
        raw_data["class"] = (
            raw_data["score_text"].map(
                {"Low": 0, "Medium": 1, "High": 2}).astype(int)
        )
        raw_data.drop(["c_jail_in", "c_jail_out", "score_text"],
                      axis=1, inplace=True)

        # Downsample to minor class
        raw_data = raw_data.dropna(subset=self.feature_columns)
        rows_per_class = raw_data[target_column].value_counts().min()
        raw_data = pd.concat(
            [
                raw_data[raw_data[target_column] == class_label].sample(
                    rows_per_class, random_state=42
                )
                for class_label in raw_data[target_column].unique()
            ]
        )

        X = raw_data[self.feature_columns].to_numpy()
        y = raw_data[target_column].to_numpy()

        return X, y
