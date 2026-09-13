import numpy as np
from tqdm import tqdm
import tensorflow as tf
from torch.utils.data import DataLoader
from alibi.explainers import Counterfactual

from counterfactuals.cf_methods.base import BaseCounterfactual, ExplanationResult
from counterfactuals.discriminative_models.base import BaseDiscModel
from hyconex.model_utils import suppress_warnings


class WACH(BaseCounterfactual):
    def __init__(
        self,
        disc_model: BaseDiscModel,
        target_class: int = "other",  # any class other than origin will do
        **kwargs,  # ignore other arguments
    ) -> None:
        tf.compat.v1.disable_eager_execution()
        self.target_proba = 1.0
        self.tol = 0.51  # want counterfactuals with p(class)>0.99
        self.target_class = target_class
        self.max_iter = 1000
        self.lam_init = 1e-1
        self.max_lam_steps = 10
        self.learning_rate_init = 0.1
        self.predict_proba = lambda x: disc_model.predict_proba(x)  # noqa: E731
        self.num_features = disc_model.input_size
        self.feature_range = (-0.5, 0.5)

        self.cf = Counterfactual(
            self.predict_proba,
            shape=(1, self.num_features),
            target_proba=self.target_proba,
            tol=self.tol,
            target_class=target_class,
            max_iter=self.max_iter,
            lam_init=self.lam_init,
            max_lam_steps=self.max_lam_steps,
            learning_rate_init=self.learning_rate_init,
            feature_range=self.feature_range,
        )

    def explain(
        self,
        X: np.ndarray,
        y_origin: np.ndarray,
        y_target: np.ndarray,
        X_train: np.ndarray,
        y_train: np.ndarray,
        **kwargs,
    ) -> ExplanationResult:
        try:
            X = X.reshape((1,) + X.shape)
            explanation = self.cf.explain(X).cf
        except Exception as e:
            explanation = None
            print(e)
        return explanation, X, y_origin, y_target
        # return ExplanationResult(
        #     x_cfs=explanation, y_cf_targets=y_target, x_origs=X, y_origs=y_origin
        # )

    @suppress_warnings
    def explain_dataloader(
        self, dataloader: DataLoader, target_class: int, *args, **kwargs
    ):
        target_value = kwargs.get("target_class_value", None)
        if target_value is not None:
            self.cf.__init__(
                self.predict_proba,
                shape=(1, self.num_features),
                target_proba=self.target_proba,
                tol=self.tol,
                target_class=target_value,
                max_iter=self.max_iter,
                lam_init=self.lam_init,
                max_lam_steps=self.max_lam_steps,
                learning_rate_init=self.learning_rate_init,
                feature_range=self.feature_range,
            )
        else:
            self.cf.__init__(
                self.predict_proba,
                shape=(1, self.num_features),
                target_proba=self.target_proba,
                tol=self.tol,
                target_class=target_class,
                max_iter=self.max_iter,
                lam_init=self.lam_init,
                max_lam_steps=self.max_lam_steps,
                learning_rate_init=self.learning_rate_init,
                feature_range=self.feature_range,
            )

        Xs, ys = dataloader.dataset.tensors
        # create ys_target numpy array same shape as ys but with target class
        Xs_cfs = []
        model_returned = []
        for X, y in tqdm(zip(Xs, ys), total=len(Xs)):
            try:
                X = X.reshape((1,) + X.shape)
                explanation = self.cf.explain(X.numpy()).cf["X"]
                model_returned.append(True)
            except Exception as e:
                explanation = np.empty_like(X.reshape(1, -1))
                explanation[:] = np.nan
                print(e)
                model_returned.append(False)
            Xs_cfs.append(explanation)

        if target_value is not None:
            ys_target = np.full(ys.shape, target_value)
        else:
            ys_target = np.abs(1-ys)

        Xs_cfs = np.array(Xs_cfs).squeeze()
        Xs = np.array(Xs)
        ys = np.array(ys)
        return Xs_cfs, Xs, ys, ys_target, model_returned
        # return ExplanationResult(x_cfs=Xs_cfs, y_cf_targets=ys, x_origs=Xs, y_origs=ys)
