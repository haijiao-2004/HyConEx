import os
from pathlib import Path
from time import time
from typing import Optional

import wandb

import numpy as np
import pandas as pd
import torch

from counterfactuals.cf_methods.base import BaseCounterfactual
from counterfactuals.cf_methods.casebased_sace.casebased_sace import CaseBasedSACE
from counterfactuals.cf_methods.cegp.cegp import CEGP
from counterfactuals.cf_methods.cem.cem import CEM_CF
from counterfactuals.cf_methods.ppcef import PPCEF
from counterfactuals.cf_methods.wach.wach import WACH
from counterfactuals.datasets.base import AbstractDataset
from counterfactuals.losses import MulticlassDiscLoss
from counterfactuals.metrics import evaluate_cf
from hyconex import configs
from hyconex.initialization import initialization
from hyconex.model import HyConEx
from hyconex.model_utils import get_classification_metrics
from hyconex.model_utils import one_hot_encoder


def evaluation_loop(
    disc_model: HyConEx,
    target_value: Optional[int],
    method_name: str,
    method_func: BaseCounterfactual,
    cf_dataloader: torch.utils.data.DataLoader,
    dataset: AbstractDataset,
    **method_kwargs
) -> pd.DataFrame:
    print("Evaluating: ", method_name)
    time_start = time()
    X_cf, X_orig, y_orig, y_target, model_returned = method_func.explain_dataloader(
        cf_dataloader, target_class_value=target_value, **method_kwargs
    )
    run_time = time() - time_start
    X_cf = one_hot_encoder(X_cf, dataset)
    if target_value is None:
        y_target = np.abs(1 - y_orig).astype(int)

    metrics = evaluate_cf(
        disc_model=disc_model,
        gen_model=disc_model.flow,
        X_cf=X_cf,
        model_returned=model_returned,
        continuous_features=dataset.numerical_features,
        categorical_features=dataset.categorical_features,
        X_train=dataset.X_train,
        y_train=dataset.y_train,
        X_test=X_orig,
        y_test=y_orig,
        median_log_prob=disc_model.log_prob_threshold,
        y_target=y_target,
        dataset=dataset,
    )

    df = pd.DataFrame(metrics, index=[0])
    df["time"] = run_time
    df.insert(0, "name", method_name)

    return df


if __name__ == "__main__":
    disc_model, config, dataset = initialization()

    checkpoint = torch.load(
        os.path.join(config.model_load_path), map_location=torch.device("cpu")
    )

    disc_model.model.load_state_dict(checkpoint["model"])
    disc_model.flow.load_state_dict(checkpoint["flow"])
    disc_model.log_prob_threshold = checkpoint["log_prob_threshold"]
    print("Original threshold: ", checkpoint["log_prob_threshold"])

    test_auroc, test_acc = get_classification_metrics(
        disc_model.predict_proba(torch.tensor(dataset.X_test)),
        torch.tensor(dataset.y_test),
        disc_model.nr_classes,
    )
    test_auroc, test_acc = test_auroc.item(), test_acc.item()
    print("Test auroc: ", test_auroc, " Test acc: ", test_acc)

    y_pred_test = disc_model.predict(dataset.X_test)
    dataset.y_test = y_pred_test

    cf_dataloader = dataset.eval_dataloader(
        batch_size=1024, shuffle=False, test=True)

    disc_model.input_size = dataset.X_train.shape[1]
    disc_model = disc_model.to("cpu")
    disc_model.eval()

    pandas_df = []

    methods = []
    if config.full_evaluation:
        methods.append(
            (
                (
                    "CEGP",
                    CEGP(
                        disc_model=disc_model,
                        beta=0.01,
                        c_init=1.0,
                        c_steps=5,
                        max_iterations=500,
                        cat_vars=(
                            {
                                class_idxs[0]: len(class_idxs)
                                for class_idxs in dataset.categorical_features_lists
                                if class_idxs
                            }
                            if len(dataset.categorical_features) != 0
                            else None
                        ),
                        num_feats=len(dataset.numerical_features),
                    ),
                    {"target_class": 0},
                )
            )
        )
        methods.append(
            (
                "CBCE",
                CaseBasedSACE(
                    disc_model=disc_model,
                    variable_features=dataset.numerical_features
                    + dataset.categorical_features,
                    continuous_features=dataset.numerical_features,
                    categorical_features_lists=dataset.categorical_features_lists,
                ),
                {
                    "X_train": dataset.X_train,
                    "y_train": dataset.y_train,
                },
            )
        )

        methods.append(
            (
                "WACH",
                WACH(
                    disc_model=disc_model,
                ),
                {"target_class": 0},
            )
        )
        methods.append(
            (
                "CEM",
                CEM_CF(
                    disc_model=disc_model,
                    mode="PN",
                    kappa=0.2,
                    beta=0.1,
                    c_init=10.0,
                    c_steps=5,
                    max_iterations=200,
                    learning_rate_init=0.01,
                ),
                {"target_class": 0},
            )
        )

        methods.append(
            (
                "PPCEF",
                PPCEF(
                    gen_model=disc_model.flow.to("cpu"),
                    disc_model=disc_model,
                    disc_model_criterion=MulticlassDiscLoss(),
                    neptune_run=None,
                ),
                {
                    "alpha": 100,
                    "epochs": 10000,
                    "log_prob_threshold": disc_model.log_prob_threshold,
                },
            )
        )

    methods_other = [
        ("HyConEx", disc_model, {"use_distance": config.use_distance}),
        (
            "HyConEx-steps",
            disc_model,
            {
                "use_distance": config.use_distance,
                "multiple_steps": True,
                "last_step_full": config.last_step_full,
                "eps": config.eps,
            },
        ),
    ]

    methods.extend(methods_other)

    for method_name, method_model, method_kwargs in methods:
        if disc_model.nr_classes == 2:
            df = evaluation_loop(
                disc_model,
                None,
                method_name,
                method_model,
                cf_dataloader,
                dataset,
                **method_kwargs
            )
            pandas_df.append(df)
        else:
            X_test_true, y_test_true = dataset.X_test, dataset.y_test
            for i in range(disc_model.nr_classes):
                mask = y_test_true != i
                dataset.X_test = X_test_true[mask]
                dataset.y_test = y_test_true[mask]
                cf_dataloader = dataset.eval_dataloader(
                    batch_size=1024, shuffle=False, test=True
                )
                dataset.X_test, dataset.y_test = X_test_true, y_test_true

                df = evaluation_loop(
                    disc_model,
                    i,
                    method_name,
                    method_model,
                    cf_dataloader,
                    dataset,
                    **method_kwargs
                )
                pandas_df.append(df)

    df = pd.concat(pandas_df, ignore_index=True)
    df = df[configs.subset_columns]

    with pd.option_context("display.max_columns", None, "display.max_rows", None):
        print(df)
        df = df.groupby("name").mean().reset_index()
        print(df)

    df["load-path"] = config.model_load_path
    df["test-auroc"] = test_auroc
    df["test-acc"] = test_acc

    if not os.path.exists("test_results.csv"):
        path = Path("test_results.csv")
        path.touch()

    df_saved = (
        pd.read_csv("test_results.csv")
        if os.path.getsize("test_results.csv") > 0
        else pd.DataFrame()
    )
    df_combined = pd.concat([df_saved, df], ignore_index=True)
    df_combined.to_csv("test_results.csv", index=False, float_format="%.3f")

    wandb.finish()
