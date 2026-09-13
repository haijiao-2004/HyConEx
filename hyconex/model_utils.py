import os
import random
import warnings
from time import time
from typing import Tuple, Union

import numpy as np
import pandas as pd
import torch
from torcheval.metrics.functional import multiclass_auroc, multiclass_accuracy

from counterfactuals.cf_methods.base import BaseCounterfactual
from counterfactuals.datasets.base import AbstractDataset
from counterfactuals.metrics import evaluate_cf
from hyconex import configs


def get_classification_metrics(predictions: torch.Tensor, y_true: torch.Tensor, nr_classes: int
                               ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute classification metrics - AUROC, accuracy for the given predictions.

    Parameters:
        predictions (torch.Tensor): The predicted class probabilities.

        y_true (torch.Tensor): The true class labels corresponding to the predictions.

        nr_classes (int): The number of distinct classes in the classification problem.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
        AUROC and accuracy of given predictions.
    """

    auroc = multiclass_auroc(predictions, y_true, num_classes=nr_classes)
    acc = multiclass_accuracy(predictions, y_true, num_classes=nr_classes)

    return auroc, acc


def eval_counterfactuals(model: Union[torch.nn.Module, BaseCounterfactual], dataset: AbstractDataset,
                         use_distance: bool, y_val_true: torch.Tensor
                         ) -> Tuple[pd.DataFrame, torch.Tensor, torch.Tensor]:
    """
    Evaluate counterfactual explanations for the given model.

    Parameters:
        model (Union[torch.nn.Module, BaseCounterfactual]): The model used to generate predictions
        and counterfactuals. This can be a PyTorch

        dataset (AbstractDataset): The dataset used to generate counterfactuals.

        use_distance (bool): Whether to use distance factor during counterfactual creation.

        y_val_true (torch.Tensor): The true labels for the validation set.

    Returns:
        Tuple[pd.DataFrame, torch.Tensor, torch.Tensor]:
        A tuple containing:
            - A pandas DataFrame with the evaluation results for the counterfactuals.
            - AUROC of validation set.
            - accuracy of validation set.
    """

    model.eval()
    eval_predictions = model.predict_proba(torch.tensor(dataset.X_val)).to(model.dev)

    auroc, acc = get_classification_metrics(
        eval_predictions, y_val_true.to(model.dev), model.nr_classes
    )
    print("Eval acc: ", acc)
    print("Eval aucroc: ", auroc)

    y_pred_val = model.predict(dataset.X_val)
    dataset.y_val = y_pred_val

    model.flow = model.flow.to("cpu")
    cf_dataloader = dataset.eval_dataloader(
        batch_size=1024, shuffle=False, test=False)

    dfs = []
    for i in range(model.nr_classes):
        time_start = time()
        X_cf, X_orig, y_orig, y_target, _ = model.explain_dataloader(
            cf_dataloader, target_class_value=i, use_distance=use_distance
        )
        run_time = time() - time_start
        X_cf = one_hot_encoder(X_cf, dataset)

        try:
            metrics = evaluate_cf(
                disc_model=model,
                gen_model=model.flow,
                X_cf=X_cf,
                model_returned=np.ones(X_cf.shape[0]),
                continuous_features=dataset.numerical_features,
                categorical_features=dataset.categorical_features,
                X_train=dataset.X_train,
                y_train=dataset.y_train,
                X_test=X_orig,
                y_test=y_orig,
                median_log_prob=model.log_prob_threshold,
                y_target=y_target,
                dataset=dataset
            )

            df = pd.DataFrame(metrics, index=[0])
            df["time"] = run_time
            df.insert(0, "name", "HyConEx")
            dfs.append(df)
        except ValueError as e:
            print(e)

    df = pd.concat(dfs, ignore_index=True)
    model.flow = model.flow.to(model.dev)

    print(df[configs.subset_columns])

    return df, auroc, acc


def pretraining_final_steps(
        model: Union[torch.nn.Module, BaseCounterfactual], dataset: AbstractDataset,
        optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.SequentialLR
):
    """
    Perform the final steps of pretraining for the given model.

    Parameters:
        model (Union[torch.nn.Module, BaseCounterfactual]): The model being pretrained.

        dataset (AbstractDataset): The dataset used for pretraining.

        optimizer (torch.optim.Optimizer): The optimizer used to update model parameters during pretraining.

        scheduler (torch.optim.lr_scheduler.SequentialLR): The learning rate scheduler used to adjust the learning rate
        throughout the pretraining process.
    """

    model.eval()
    y_pred_train = model.predict(torch.tensor(dataset.X_train).to(model.dev))
    y_pred_val = model.predict(torch.tensor(dataset.X_val).to(model.dev))

    dataset.y_train = y_pred_train.detach().cpu().numpy()
    dataset.y_val = y_pred_val.detach().cpu().numpy()

    eval_dataloader = dataset.eval_dataloader(
        batch_size=128, shuffle=False, test=False)

    model.flow = model.flow.to("cpu")
    model.flow.fit(
        dataset.train_dataloader(batch_size=256, shuffle=True),
        eval_dataloader,
        num_epochs=10000,
        patience=50,
        learning_rate=5e-4,
    )
    log_prob_threshold = torch.quantile(
        model.flow.predict_log_prob(
            dataset.train_dataloader(batch_size=1024, noise_factor=0., cat_noise_factor=0., shuffle=False)
        ),
        0.25,
    )
    print("Log prob threshold ", log_prob_threshold)

    checkpoint = {
        "model": model.model.state_dict(),
        "flow": model.flow.state_dict(),
        "log_prob_threshold": log_prob_threshold,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": model.config
    }
    torch.save(checkpoint, os.path.join(
        model.output_directory, "checkpoint.pt"))


def set_seed(seed: int):
    """
    Set the seed for reproducibility across Python, NumPy, and PyTorch.

    Parameters:
        seed (int): The seed value to be used for random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def one_hot_encoder(X_cf: np.ndarray, dataset: AbstractDataset) -> np.ndarray:
    """
    Transform categorical features in continuous representation back to a one-hot representation.

    Parameters:
        X_cf (np.ndarray): Counterfactual examples with categorical features in continuous representation.

        dataset (AbstractDataset): The dataset object that provides information about categorical feature mappings.

    Returns:
        np.ndarray: An array containing the one-hot encoded representation of the input data.
    """

    if len(dataset.categorical_features) == 0:
        return X_cf

    ohe = dataset.feature_transformer.transformers_[1][1]
    categories = ohe.categories_

    start_idx = len(dataset.numerical_columns)
    for i, cat in enumerate(categories):
        cls = len(cat)
        predicted_cat = np.argmax(X_cf[:, start_idx: start_idx + cls], axis=-1)
        one_hot_encoded = np.eye(cls)[predicted_cat]
        X_cf[:, start_idx: start_idx + cls] = one_hot_encoded
        start_idx += cls
    return X_cf


def one_hot_encoder_torch(X_cf: torch.Tensor, dataset: AbstractDataset, T: float) -> torch.Tensor:
    """
    Transform categorical features in continuous representation to a soft one-hot representation
    using softmax with temperature.

    Parameters:
        X_cf (torch.Tensor): Counterfactual examples with categorical features in continuous representation.

        dataset (AbstractDataset): The dataset object that provides information about categorical feature mappings.

        T (float): softmax temperature.

    Returns:
        torch.Tensor: An array containing the one-hot encoded representation of the input data.
    """

    if len(dataset.categorical_features) == 0:
        return X_cf
    ohe = dataset.feature_transformer.transformers_[1][1]
    categories = ohe.categories_

    start_idx = len(dataset.numerical_columns)
    for i, cat in enumerate(categories):
        cls = len(cat)
        one_hot_soft = torch.softmax(
            X_cf[:, start_idx: start_idx + cls] / T, dim=-1)
        X_cf[:, start_idx: start_idx + cls] = one_hot_soft
        start_idx += cls
    return X_cf


def suppress_warnings(func):
    """Decorator to suppress warnings for a specific function."""
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return func(*args, **kwargs)
    return wrapper
