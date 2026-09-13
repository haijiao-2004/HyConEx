import numpy as np
import pandas as pd
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from counterfactuals.datasets.base import AbstractDataset


class Moons3Dataset(AbstractDataset):
    def __init__(self):
        super().__init__()
        X, y = make_moons(n_samples=1400, noise=0.07, random_state=42)

        X[y == 0] = X[y == 0] + [-0.3, 0.3]
        X_extra = X[y == 0] + [2.6, 0.0]
        y_extra = np.full(X_extra.shape[0], 2)
        X[y == 1] = 1.2 * X[y == 1] - np.array([0.2, 0.0])

        X = np.vstack([X, X_extra])
        y = np.hstack([y, y_extra])
        X[:, 1] = -X[:, 1]

        self.X, self.y = X, y

        self.preprocess(pd.DataFrame())

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

    def preprocess(self, raw_data: pd.DataFrame):
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

        self.numerical_columns = [0, 1]
        self.categorical_columns = []