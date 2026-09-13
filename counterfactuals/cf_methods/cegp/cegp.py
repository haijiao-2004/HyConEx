import numpy as np
from tqdm import tqdm
import tensorflow as tf
from torch.utils.data import DataLoader
from alibi.explainers import CounterFactualProto

from counterfactuals.cf_methods.base import BaseCounterfactual, ExplanationResult
from counterfactuals.discriminative_models.base import BaseDiscModel


class CEGP(BaseCounterfactual):
    def __init__(
        self,
        disc_model: BaseDiscModel,
        beta: float = 0.01,
        c_init: float = 1.0,
        c_steps: int = 5,
        max_iterations: int = 500,
        cat_vars=None,
        num_feats=0,
        **kwargs,  # ignore other arguments
    ) -> None:
        """Initialize CEGP counterfactual method.

        Args:
            disc_model: Discriminative model to use for counterfactual generation
            beta: Trade-off parameter for distance computation
            c_init: Initial value of c for the attack loss term
            c_steps: Number of steps to adjust c
            max_iterations: Maximum number of iterations to run optimization
        """
        tf.compat.v1.disable_eager_execution()

        def predict_proba(x):
            return disc_model.predict_proba(x)  # noqa: E731

        features = disc_model.input_size
        shape = (1, features)

        # Get feature ranges from training data
        # Default range, should be adjusted based on dat
        if cat_vars is None:
            feature_range = (-0.5, 0.5)
        else:
            cat_feats = len(cat_vars.keys())
            num_feature_range = (
                -0.5 * np.ones((1, num_feats)),
                0.5 * np.ones((1, num_feats)),
            )
            cat_feature_range = (-np.ones((1, cat_feats)),
                                 np.ones((1, cat_feats)))
            feature_range = (
                np.hstack((num_feature_range[0], cat_feature_range[0])),
                np.hstack((num_feature_range[1], cat_feature_range[1])),
            )
        print(cat_vars)

        self.cf = CounterFactualProto(
            predict_proba,
            shape,
            beta=beta,
            max_iterations=max_iterations,
            feature_range=feature_range,
            c_init=c_init,
            c_steps=c_steps,
            ohe=True,
            cat_vars=cat_vars,
        )
        self.is_fitted = False

    def fit(self, X_train: np.ndarray):
        """Fit the CEGP model on training data.

        Args:
            X_train: Training data to fit the model on
        """
        self.cf.fit(X_train.astype(np.float32),
                    d_type="abdm", disc_perc=[25, 50, 75])
        self.is_fitted = True

    def explain(
        self,
        X: np.ndarray,
        y_origin: np.ndarray,
        y_target: np.ndarray,
        X_train: np.ndarray,
        y_train: np.ndarray,
        **kwargs,
    ) -> ExplanationResult:
        """Generate counterfactual explanations for given samples.

        Args:
            X: Samples to explain
            y_origin: Original labels
            y_target: Target labels
            X_train: Training data (used for fitting if not already fitted)
            y_train: Training labels
        """
        if not self.is_fitted:
            self.fit(X_train)

        try:
            X = X.reshape((1,) + X.shape)
            explanation = self.cf.explain(X).cf
            if explanation is None:
                raise ValueError("No counterfactual found")
        except Exception as e:
            explanation = None
            print(f"Error in CEGP explanation: {e}")

        return explanation, X, y_origin, y_target

    def explain_dataloader(
        self, dataloader: DataLoader, target_class: int, *args, **kwargs
    ) -> ExplanationResult:
        """Generate counterfactual explanations for all samples in dataloader.

        Args:
            dataloader: DataLoader containing samples to explain
            target_class: Target class for counterfactuals
        """
        Xs, ys = dataloader.dataset.tensors

        # Fit on first batch if not already fitted
        if not self.is_fitted:
            self.fit(Xs.numpy())

        # create ys_target numpy array same shape as ys but with target class
        target_value = kwargs.get("target_class_value", None)
        if target_value is not None:
            target_value = [target_value]

        Xs_cfs = []
        model_returned = []

        for X, y in tqdm(zip(Xs, ys), total=len(Xs)):
            try:
                X = X.reshape((1,) + X.shape)
                explanation = self.cf.explain(
                    X.numpy(), target_class=target_value).cf
                if explanation is None:
                    raise ValueError("No counterfactual found")
                Xs_cfs.append(explanation["X"])
                model_returned.append(True)
            except Exception as e:
                print(f"Error in CEGP explanation: {e}")
                explanation = np.empty_like(X.reshape(1, -1))
                explanation[:] = np.nan
                Xs_cfs.append(explanation)
                model_returned.append(False)

        if target_value is not None:
            ys_target = np.full(ys.shape, target_value)
        else:
            ys_target = np.abs(1 - ys)

        Xs_cfs = np.array(Xs_cfs).squeeze()
        Xs = np.array(Xs)
        ys = np.array(ys)
        ys_target = np.array(ys_target)
        return Xs_cfs, Xs, ys, ys_target, model_returned
