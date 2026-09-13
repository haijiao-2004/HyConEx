from typing import Optional, List, Union, cast
import logging

import torch
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OrdinalEncoder

from counterfactuals.datasets.base import AbstractDataset
from counterfactuals.metrics.distances import (
    continuous_distance,
    categorical_distance,
    distance_combined,
)


logger = logging.getLogger(__name__)


def _convert_to_numpy(X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Convert input data to numpy array.

    Args:
        X (np.ndarray | torch.Tensor): Input data.

    Returns:
        np.ndarray: Converted array.

    Raises:
        ValueError: If X is neither a numpy array nor a torch tensor.
    """
    if isinstance(X, np.ndarray):
        return X
    elif isinstance(X, torch.Tensor):
        return X.detach().numpy()
    else:
        raise ValueError("X should be either a numpy array or a torch tensor")


class CFMetrics:
    """
    Class for computing counterfactual metrics.
    Args:
        X_cf (np.ndarray | torch.Tensor): Counterfactual instances.
        y_target (np.ndarray | torch.Tensor): Target labels for counterfactual instances.
        X_train (np.ndarray | torch.Tensor): Training instances.
        y_train (np.ndarray | torch.Tensor): Training labels.
        X_test (np.ndarray | torch.Tensor): Test instances.
        y_test (np.ndarray | torch.Tensor): Test labels.
        gen_model (torch.nn.Module): Generator model.
        disc_model (torch.nn.Module): Discriminator model.
        continuous_features (list[int]): List of indices of continuous features.
        categorical_features (list[int]): List of indices of categorical features.
        ratio_cont (Optional[float], optional): Ratio of continuous features to be perturbed. Defaults to None.
        prob_plausibility_threshold (Optional[float], optional): Log Likelihood Threshold for prob. plausibility.
        Defaults to None.
    """

    def __init__(
        self,
        X_cf: Union[np.ndarray, torch.Tensor],
        y_target: Union[np.ndarray, torch.Tensor],
        X_train: Union[np.ndarray, torch.Tensor],
        y_train: Union[np.ndarray, torch.Tensor],
        X_test: Union[np.ndarray, torch.Tensor],
        y_test: Union[np.ndarray, torch.Tensor],
        gen_model: torch.nn.Module,
        disc_model: torch.nn.Module,
        continuous_features: List[int],
        categorical_features: List[int],
        ratio_cont: Optional[float] = None,
        prob_plausibility_threshold: Optional[float] = None,
        *,
        model_returned,
        dataset
    ) -> None:
        # precheck input assumptions
        assert (
            X_cf.shape[1] == X_train.shape[1] == X_test.shape[1]
        ), "All input data should have the same number of features"
        assert (
            X_train.shape[0] == y_train.shape[0]
        ), "X_train and y_train should have the same number of samples"
        assert (
            X_test.shape[0] == y_test.shape[0]
        ), "X_test and y_test should have the same number of samples"
        assert (
            X_cf.shape[0] == y_test.shape[0]
        ), "X_cf and y_test should have the same number of samples"
        assert (
            len(continuous_features) +
            len(categorical_features) == X_cf.shape[1]
        ), "The sum of continuous and categorical features should equal the number of features in X_cf"
        assert (
            ratio_cont is None or 0 <= ratio_cont <= 1
        ), "ratio_cont should be between 0 and 1"

        # convert everything to torch tensors if not already
        self.X_cf = _convert_to_numpy(X_cf)
        self.y_target = _convert_to_numpy(np.squeeze(y_target))
        self.X_train = _convert_to_numpy(X_train)
        self.y_train = _convert_to_numpy(y_train)
        self.X_test = _convert_to_numpy(X_test)
        self.y_test = _convert_to_numpy(y_test)

        # write class properties
        self.gen_model = gen_model
        self.disc_model = disc_model
        self.prob_plausibility_threshold = (
            prob_plausibility_threshold.item()
            if isinstance(prob_plausibility_threshold, torch.Tensor)
            else prob_plausibility_threshold
        )

        # set models to evaluation mode
        self.gen_model = self.gen_model.eval()
        self.disc_model = self.disc_model.eval()

        self.continuous_features = continuous_features
        self.categorical_features = categorical_features
        self.ratio_cont = ratio_cont

        # filter only valid counterfactuals and test instances
        self.y_cf_pred = _convert_to_numpy(self.disc_model.predict(self.X_cf))
        self.X_cf_valid = self.X_cf[self.y_cf_pred == self.y_target]
        self.X_test_valid = self.X_test[self.y_cf_pred == self.y_target]
        print(self.y_cf_pred.shape)
        print(self.X_cf_valid.shape)
        self.model_returned = model_returned

        # revert one-hots
        if len(categorical_features) != 0 and len(self.X_cf_valid) != 0:

            ord_enc = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )

            ord_enc.fit_transform(
                dataset.feature_transformer.named_transformers_[
                    "OneHotEncoder"
                ].inverse_transform(self.X_train[:, dataset.categorical_features])
            )
            self.X_test_valid[:, dataset.categorical_columns] = ord_enc.transform(
                dataset.feature_transformer.named_transformers_[
                    "OneHotEncoder"
                ].inverse_transform(self.X_test_valid[:, dataset.categorical_features])
            )
            self.X_cf_valid[:, dataset.categorical_columns] = ord_enc.transform(
                dataset.feature_transformer.named_transformers_[
                    "OneHotEncoder"
                ].inverse_transform(self.X_cf_valid[:, dataset.categorical_features])
            )

            init_features = len(dataset.numerical_columns) + len(
                dataset.categorical_columns
            )
            self.X_cf_valid = self.X_cf_valid[:, :init_features]
            self.X_test_valid = self.X_test_valid[:, :init_features]

            self.categorical_features = dataset.categorical_columns

    def coverage(self) -> float:
        """
        Compute the coverage metric.

        Returns:
            float: Coverage metric value.
        """
        # check how many vectors of dim 0 contain NaN in X_cf
        # return 1 - np.isnan(self.X_cf).any(axis=1).mean()
        return np.sum(self.model_returned) / len(self.model_returned)

    def validity(self) -> float:
        """
        Compute the validity metric.

        Returns:
            float: Validity metric value.
        """
        return (cast(np.ndarray, self.y_cf_pred == self.y_target)).mean()

    def actionability(self) -> float:
        """
        Compute the actionability metric.

        Returns:
            float: Actionability metric value.
        """
        return np.all(self.X_test == self.X_cf, axis=1).mean()

    def sparsity(self) -> float:
        """
        Compute the sparsity metric.

        Returns:
            float: Sparsity metric value.
        """
        return (cast(np.ndarray, self.X_test != self.X_cf)).mean()

    def prob_plausibility(self, cf: bool = True, *, threshold: float) -> float:
        """
        Compute the probability plausibility metric.
        This metric is computed as the average number of counterfactuals that are more plausible than the threshold.
        Args:
            cf (bool, optional): Whether to compute the metric for counterfactuals of for X_test. Defaults to True.
            threshold: Plausibility threshold

        Returns:
            float: Avg number of counterfactuals that are more plausible than the threshold.
        """
        X = self.X_cf if cf else self.X_test
        X = torch.from_numpy(X).float()
        y = torch.from_numpy(self.y_target).float()
        gen_log_probs: np.ndarray = self.gen_model(X, y).detach().numpy()
        return (gen_log_probs > threshold).mean()

    def log_density(self, cf: bool = True) -> float:
        """
        Compute the log density metric.
        This metric is computed as the average log density of the counterfactuals.
        Args:
            cf (bool, optional): Whether to compute the metric for counterfactuals of for X_test. Defaults to True.

        Returns:
            float: Average log density of the counterfactuals.
        """
        X = self.X_cf if cf else self.X_test
        X = torch.from_numpy(X).float()
        y = torch.from_numpy(self.y_target).float()
        gen_log_probs = self.gen_model(X, y).detach().numpy()
        return np.median(gen_log_probs)

    def lof_scores(self, cf: bool = True, n_neighbors: int = 20) -> float:
        """
        Compute the Local Outlier Factor (LOF) metric.
        This metric is computed as the average LOF score of the counterfactuals.
        LOF(k) ~ 1 means Similar density as neighbors,

        LOF(k) < 1 means Higher density than neighbors (Inlier),

        LOF(k) > 1 means Lower density than neighbors (Outlier)
        Args:
            cf (bool, optional): Whether to compute the metric for counterfactuals of for X_test. Defaults to True.
            n_neighbors (int, optional): Number of neighbors to consider. Defaults to 20.

        Returns:
            float: Average LOF score of the counterfactuals.
        """
        X = self.X_cf if cf else self.X_test
        X_train = self.X_train

        lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
        lof.fit(X_train)

        # ORIG: It is the opposite as bigger is better, i.e. large values correspond to inliers.
        # NEG: smaller is better, i.e. small values correspond to inliers.
        lof_scores = -lof.score_samples(X)
        return np.median(lof_scores)

    def isolation_forest_scores(
        self, cf: bool = True, n_estimators: int = 100
    ) -> float:
        """
        Compute the Isolation Forest metric.
        This metric is computed as the average Isolation Forest score of the counterfactuals.
        The score is between -0.5 and 0.5, where smaller values mean more anomalous.
        https://stackoverflow.com/questions/45223921/what-is-the-range-of-scikit-learns-isolationforest-decision-function-scores#51882974

        The anomaly score of the input samples. The lower, the more abnormal.
        Negative scores represent outliers, positive scores represent inliers.
        See: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html#sklearn.ensemble.
        IsolationForest.decision_function

        Args:
            cf (bool, optional): Whether to compute the metric for counterfactuals of for X_test. Defaults to True.
            n_estimators (int, optional): Number of trees in the forest. Defaults to 100.

        Returns:
            float: Average Isolation Forest score of the counterfactuals.
        """
        X = self.X_cf if cf else self.X_test
        X_train = self.X_train

        clf = IsolationForest(n_estimators=n_estimators, random_state=42)
        clf.fit(X_train)
        lof_scores = clf.decision_function(X)
        return lof_scores.mean()

        # isolation_forest_scores = isolation_forest_metric(X_train, X, self.X_test, n_estimators) TODO: fix this
        # return isolation_forest_scores.mean()

    def feature_distance(
        self,
        continuous_metric: Optional[str] = None,
        categorical_metric: Optional[str] = None,
        X_train: Optional[np.ndarray] = None,
    ) -> float:
        X = self.X_cf_valid
        X_test = self.X_test_valid

        if not any([continuous_metric, categorical_metric]):
            raise ValueError(
                "At least one of continuous_metric or categorical_metric should be provided"
            )
        elif categorical_metric is None:
            return continuous_distance(
                X_test=X_test,
                X_cf=X,
                continuous_features=self.continuous_features,
                metric=continuous_metric,
                X_all=X_train,
                agg="mean",
            )
        elif continuous_metric is None:
            return categorical_distance(
                X_test=X_test,
                X_cf=X,
                categorical_features=self.categorical_features,
                metric=categorical_metric,
                agg="mean",
            )
        else:
            return distance_combined(
                X_test=X_test,
                X_cf=X,
                X_all=X_train,
                continuous_metric=continuous_metric,
                categorical_metric=categorical_metric,
                continuous_features=self.continuous_features,
                categorical_features=self.categorical_features,
                ratio_cont=self.ratio_cont,
            )

    def target_distance(self, metric: str = "euclidean") -> float:
        """
        Compute the distance metric between targets (used for regression setup).

        Returns:
            float: Distance metric value between targets.
        """
        return continuous_distance(
            X_test=self.y_target,
            X_cf=self.y_cf_pred,
            continuous_features=[0],
            metric=metric,
            X_all=None,
            agg="mean",
        )

    def calc_all_metrics(self) -> dict:
        """
        Calculate all metrics.

        Returns:
            dict: Dictionary of metric names and values.
        """
        half_threshold = (
            self.prob_plausibility_threshold / 2
            if self.prob_plausibility_threshold > 0
            else self.prob_plausibility_threshold * 1.5
        )
        metrics = {
            "coverage": self.coverage(),
            "validity": self.validity(),
            "actionability": self.actionability(),
            "sparsity": self.sparsity(),
            # "target_distance": self.target_distance(),
            "proximity_categorical_hamming": self.feature_distance(
                categorical_metric="hamming"
            ),
            "proximity_categorical_jaccard": self.feature_distance(
                categorical_metric="jaccard"
            ),
            "proximity_continuous_manhattan": self.feature_distance(
                continuous_metric="cityblock"
            ),
            "proximity_continuous_euclidean": self.feature_distance(
                continuous_metric="euclidean"
            ),
            "proximity_l2_jaccard": self.feature_distance(
                continuous_metric="euclidean", categorical_metric="jaccard"
            ),
            "prob_plausibility": self.prob_plausibility(
                cf=True, threshold=self.prob_plausibility_threshold
            ),
            "log_density_cf": self.log_density(cf=True),
            "log_density_test": self.log_density(cf=False),
            "lof_scores_cf": self.lof_scores(cf=True),
            "lof_scores_test": self.lof_scores(cf=False),
            "isolation_forest_scores_cf": self.isolation_forest_scores(cf=True),
            "isolation_forest_scores_test": self.isolation_forest_scores(cf=False),
        }
        return metrics


def evaluate_cf(
    disc_model: torch.nn.Module,
    gen_model: torch.nn.Module,
    X_cf: np.ndarray,
    model_returned: np.ndarray,
    continuous_features: list,
    categorical_features: list,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    median_log_prob: Union[float, None],
    y_target: Optional[np.ndarray] = None,
    *,
    dataset: AbstractDataset = None
):
    y_target = np.abs(1 - y_test) if y_target is None else y_target

    model_returned = np.array(model_returned).astype(bool)

    X_cf = X_cf[model_returned]
    X_test = X_test[model_returned]
    y_target = y_target[model_returned]
    y_test = y_test[model_returned]

    mask = cast(np.ndarray, y_target != y_test)
    if len(mask.shape) > 1:
        mask = mask.squeeze(-1)
    print("Original shape: ", X_cf.shape)
    X_cf = X_cf[mask]
    X_test = X_test[mask]
    y_target = y_target[mask]
    y_test = y_test[mask]
    print("Masked shaped: ", X_cf.shape)

    metrics_cf = CFMetrics(
        disc_model=disc_model,
        gen_model=gen_model,
        X_cf=X_cf,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        y_target=y_target,
        continuous_features=continuous_features,
        categorical_features=categorical_features,
        ratio_cont=None,
        prob_plausibility_threshold=median_log_prob,
        model_returned=model_returned,
        dataset=dataset,
    )
    metrics = metrics_cf.calc_all_metrics()
    return metrics
