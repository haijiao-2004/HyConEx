import openml
import pandas as pd
from sklearn.model_selection import train_test_split

from counterfactuals.datasets.base import AbstractDataset


class OpenmlDataset(AbstractDataset):

    def __init__(self, dataset_id):
        super().__init__("")

        dataset = openml.datasets.get_dataset(dataset_id)
        dataset_name = dataset.name
        print(dataset_name)
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            dataset_format="dataframe", target=dataset.default_target_attribute
        )

        self.X, self.y, self.attribute_names = X, y, attribute_names
        self.categorical_indicator = categorical_indicator
        self.preprocess(pd.DataFrame())

        self.raw_data = self.X_train
        self.feature_columns = self.attribute_names

        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = (
            self.transform(
                self.X_train,
                self.X_val,
                self.X_test,
                self.y_train,
                self.y_val,
                self.y_test,
            )
        )

        print(
            f"Train len {len(self.X_train)} Val len {len(self.X_val)} Test val {len(self.X_test)}"
        )
        print("Categorical features: ", self.categorical_features)

    def preprocess(self, raw_data: pd.DataFrame):
        dropped_column_names = []
        dropped_column_indices = []

        for column_index, column_name in enumerate(self.X.keys()):
            if self.X[column_name].isnull().sum() > len(self.X[column_name]) * 0.9:
                dropped_column_names.append(column_name)
                dropped_column_indices.append(column_index)
            if self.X[column_name].nunique() == 1:
                dropped_column_names.append(column_name)
                dropped_column_indices.append(column_index)

        for column_index, column_name in enumerate(self.X.keys()):
            if (
                    self.X[column_name].dtype == "object"
                    or self.X[column_name].dtype == "category"
                    or self.X[column_name].dtype == "string"
            ):
                if self.X[column_name].nunique() / len(self.X[column_name]) > 0.9:
                    dropped_column_names.append(column_name)
                    dropped_column_indices.append(column_index)

        self.X = self.X.drop(dropped_column_names, axis=1)
        self.categorical_indicator = [
            self.categorical_indicator[i]
            for i in range(len(self.categorical_indicator))
            if i not in dropped_column_indices
        ]
        self.attribute_names = [
            attribute_name
            for attribute_name in self.attribute_names
            if attribute_name not in dropped_column_names
        ]

        column_category_values = []
        # take pandas categories into account
        for cat_indicator, column_name in zip(
            self.categorical_indicator, self.X.keys()
        ):
            if cat_indicator:
                column_categories = list(self.X[column_name].cat.categories)
                column_category_values.append(column_categories)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=0.2,
            random_state=4,
            stratify=self.y,
        )

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_train,
            self.y_train,
            test_size=0.2,
            random_state=4,
            stratify=self.y_train,
        )

        self.numerical_columns = [
            i
            for i in range(len(self.categorical_indicator))
            if not self.categorical_indicator[i]
        ]
        self.categorical_columns = [
            i
            for i in range(len(self.categorical_indicator))
            if self.categorical_indicator[i]
        ]
