from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold

import torch
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    LabelEncoder,
)
from torch.utils.data import DataLoader, TensorDataset


class AbstractDataset(ABC):
    """Abstract dataset class."""
    def __init__(self, data=None):
        self.y_val = None
        self.data = data
        self.numerical_features = []
        self.numerical_columns = []
        self.categorical_features = []
        self.categorical_columns = []
        self.actionable_features = []

    def _normal_init(self, file_path, **kwargs):
        self.raw_data = (
            self.load(file_path=file_path, **kwargs)
            if file_path is not None
            else self.raw_data
        )
        self.X, self.y = self.preprocess(raw_data=self.raw_data)
        self.X_train, self.X_test, self.y_train, self.y_test = self.get_split_data(
            self.X, self.y
        )
        self.X_train, self.X_val, self.y_train, self.y_val = self.get_split_data(
            self.X_train, self.y_train, state=0
        )

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

    @abstractmethod
    def preprocess(self, raw_data: pd.DataFrame):
        """
        Preprocess the loaded data, if necessary.
        """
        pass

    def load(self, file_path, **kwargs) -> pd.DataFrame:
        """
        Load data from a CSV file and store it in the 'data' attribute.
        """

        try:
            data = pd.read_csv(file_path, **kwargs)
            return data
        except Exception as e:
            raise FileNotFoundError(
                f"Error loading data from {file_path}: {e}")

    def save(self, file_path: str, data: pd.DataFrame, **kwargs):
        """
        Save the processed data (including scaled features) to a CSV file.
        """

        if data is not None:
            try:
                data.to_csv(file_path, **kwargs)
                print(f"Data saved to {file_path}")
            except Exception as e:
                print(f"Error saving data to {file_path}: {e}")
        else:
            print("No data to save.")

    def get_cv_splits(self, n_splits: int = 5):
        """
        Sets and return the train and test splits for cross-validation.
        """

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=4)
        for train_idx, test_idx in cv.split(self.X, self.y):
            self.X_train, self.X_test = self.X[train_idx], self.X[test_idx]
            self.y_train, self.y_test = self.y[train_idx], self.y[test_idx]
            (
                self.X_train,
                self.X_val,
                self.X_test,
                self.y_train,
                self.y_val,
                self.y_test,
            ) = self.transform(
                X_train=self.X_train,
                X_val=self.X_val,
                X_test=self.X_test,
                y_train=self.y_train,
                y_val=self.y_val,
                y_test=self.y_test,
            )
            yield self.X_train, self.X_test, self.y_train, self.y_test

    def get_split_data(self, X: np.ndarray, y: np.ndarray, state=4):
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            random_state=state,
            test_size=0.2,
            shuffle=True,
            stratify=y,
        )
        return X_train, X_test, y_train, y_test

    def train_dataloader(
        self,
        batch_size: int,
        shuffle: bool,
        noise_factor=0.2,
        cat_noise_factor=0.1,
        pretrain=False,
        y_target=None,
        **kwargs_dataloader
    ) -> torch.utils.data.DataLoader:
        """Create train dataloader."""

        print("Pretrain: ", pretrain)
        self.X_noise = torch.tensor(
            [
                1 / len(np.unique(self.X_train[:, i])) * noise_factor
                for i in self.numerical_features
            ]
        )

        class MinDateset(torch.utils.data.Dataset):
            def __init__(self, X: np.ndarray, y: np.ndarray, y_target=None):
                self.min_targets = []

                y_t = y_target if y_target is not None else y
                unique_classes = np.unique(y_t)
                # dists = cdist(X, X)
                for c in unique_classes:
                    idx_c = np.where(y_t == c)[0]
                    X_c = X[idx_c]
                    min_clusters = min(int(X_c.shape[0] / 5), 25)
                    n_clusters = max(min_clusters, min(len(X_c) // 20, 100))
                    kmeans = KMeans(n_clusters=n_clusters,
                                    random_state=0, n_init=10)
                    kmeans.fit(X_c)
                    dists = cdist(X, kmeans.cluster_centers_)

                    min_poses = np.argmin(dists, axis=-1, keepdims=True)
                    min_X_c = kmeans.cluster_centers_[min_poses]
                    self.min_targets.append(min_X_c)

                self.min_targets = torch.from_numpy(
                    np.concatenate(self.min_targets, axis=1)
                )
                self.X = torch.from_numpy(X)
                self.y = torch.from_numpy(y)
                print(X.shape[1])
                print("Targets: ", self.min_targets.shape)

            def __len__(self):
                return len(self.X)

            def __getitem__(self, idx):
                return self.X[idx], self.y[idx], self.min_targets[idx]

        def collate_fn(batch):
            if not pretrain:
                X, y = zip(*batch)
            else:
                X, y, z = zip(*batch)
                z = torch.stack(z)
            X = torch.stack(X)
            y = torch.stack(y)

            # Add Gaussian noise to train features
            if noise_factor != 0:
                noise_level = self.X_noise
                noise = torch.randn_like(
                    X[:, self.numerical_features]) * noise_level
                X[:, self.numerical_features] = X[:,
                                                  self.numerical_features] + noise

            if cat_noise_factor != 0:
                noise = (
                    torch.randn_like(
                        X[:, self.categorical_features]) * cat_noise_factor
                )
                X[:, self.categorical_features] = (
                    X[:, self.categorical_features] + noise
                )

            if not pretrain:
                return X, y
            else:
                return X, y, z

        if not pretrain:
            dataset = TensorDataset(
                torch.from_numpy(self.X_train), torch.from_numpy(self.y_train)
            )
        else:
            dataset = MinDateset(self.X_train, self.y_train, y_target)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn if noise_factor or cat_noise_factor else None,
            **kwargs_dataloader,
        )

    def eval_dataloader(
        self, batch_size: int, shuffle: bool, *, test, **kwargs_dataloader
    ) -> torch.utils.data.DataLoader:
        """Create evaluation dataloader."""

        if test:
            X, y = self.X_test, self.y_test
        else:
            X, y = self.X_val, self.y_val

        return DataLoader(
            TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
            batch_size=batch_size,
            shuffle=shuffle,
            **kwargs_dataloader,
        )

    @property
    def categorical_features_lists(self) -> list:
        categorical_features_lists = []
        for col in self.categorical_columns:
            n_cat = self.raw_data[self.feature_columns[col]].nunique()
            if len(categorical_features_lists) == 0:
                categorical_features_lists.append(
                    list(
                        range(
                            len(self.numerical_columns),
                            len(self.numerical_columns) + n_cat,
                        )
                    )
                )
            else:
                categorical_features_lists.append(
                    list(
                        range(
                            categorical_features_lists[-1][-1] + 1,
                            categorical_features_lists[-1][-1] + 1 + n_cat,
                        )
                    )
                )
        return categorical_features_lists

    def transform(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
    ):
        """
        Transform the loaded data by applying Min-Max scaling and OneHotEncoder to the features.
        """
        self.feature_transformer = ColumnTransformer(
            [
                (
                    "Num_scaler",
                    Pipeline(
                        steps=[("min_max", MinMaxScaler(
                            feature_range=(-0.5, 0.5)))]
                    ),
                    self.numerical_columns,
                ),
                (
                    "OneHotEncoder",
                    OneHotEncoder(handle_unknown="ignore",
                                  sparse_output=False),
                    self.categorical_columns,
                ),
            ],
        )
        X_train = self.feature_transformer.fit_transform(X_train)
        X_val = self.feature_transformer.transform(X_val)
        X_test = self.feature_transformer.transform(X_test)

        self.target_transformer = LabelEncoder()
        y_train = self.target_transformer.fit_transform(y_train)
        y_val = self.target_transformer.transform(y_val)
        y_test = self.target_transformer.transform(y_test)

        y_train = y_train.reshape(-1)
        y_val = y_val.reshape(-1)
        y_test = y_test.reshape(-1)

        X_train = X_train.astype(np.float32)
        X_val = X_val.astype(np.float32)
        X_test = X_test.astype(np.float32)
        y_train = y_train.astype(np.int64)
        y_val = y_val.astype(np.int64)
        y_test = y_test.astype(np.int64)

        self.numerical_features = list(range(0, len(self.numerical_columns)))
        self.categorical_features = list(
            range(len(self.numerical_columns), X_train.shape[1])
        )
        self.actionable_features = list(range(0, X_train.shape[1]))

        return X_train, X_val, X_test, y_train, y_val, y_test
