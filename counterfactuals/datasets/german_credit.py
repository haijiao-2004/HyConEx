import pandas as pd

from counterfactuals.datasets.base import AbstractDataset


class GermanCreditDataset(AbstractDataset):
    def __init__(self, file_path: str = "data/german_credit.csv"):
        super().__init__()
        self._normal_init(file_path, index_col=False)

    def preprocess(self, raw_data: pd.DataFrame):
        """
        Preprocess the loaded data to X and y numpy arrays.
        """

        self.feature_columns = [
            # Continuous
            "duration_in_month",
            "credit_amount",
            "installment_as_income_perc",
            "present_res_since",
            "age",
            "credits_this_bank",
            "people_under_maintenance",
            # Categorical
            "account_check_status",
            "credit_history",
            "purpose",
            "savings",
            "present_emp_since",
            "personal_status_sex",
            "other_debtors",
            "property",
            "other_installment_plans",
            "housing",
            "job",
            "telephone",
            "foreign_worker",
        ]
        self.numerical_columns = list(range(0, 7))
        self.categorical_columns = list(range(7, len(self.feature_columns)))
        target_column = "default"

        # Downsample to minor class
        print(raw_data.columns)
        raw_data = raw_data.dropna(subset=self.feature_columns)
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
