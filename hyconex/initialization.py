import os
import sys
from typing import cast, Tuple

import numpy as np
import torch
import wandb
from omegaconf import OmegaConf

from counterfactuals.datasets.base import AbstractDataset
from counterfactuals.generative_models import MaskedAutoregressiveFlow
from hyconex.configs import HyConExConfig, print_help, DatasetType
from hyconex.model import HyConEx
from hyconex.model_utils import set_seed


def config_initialization() -> HyConExConfig:
    """
    Initialize HyConExConfig.

    Returns:
        HyConExConfig: Initialized configuration
    """

    if "--help" in sys.argv:
        print_help(HyConExConfig)
        sys.exit(0)

    base_config = OmegaConf.structured(HyConExConfig)
    arg_config = OmegaConf.from_cli()

    config = cast(HyConExConfig,
                  OmegaConf.merge(base_config, arg_config))
    config.dataset_name = config.dataset.name + config.openml_id

    return config


def model_initialization(
    dataset: AbstractDataset, config: HyConExConfig
) -> HyConEx:
    """
    Initialize Classifier model.

    Parameters:
        dataset (AbstractDataset): The dataset to be used for training or inference.

        config (HyConExConfig): A configuration object containing settings specific to the model.

    Returns:
        HyConEx: Initialized model
    """

    nr_features = dataset.X_train.shape[1] if len(
        dataset.X_train.shape) > 1 else 1
    unique_classes, class_counts = np.unique(
        dataset.y_train, axis=0, return_counts=True
    )
    nr_classes = len(unique_classes)
    print("Classes ", nr_classes)

    network_configuration = {
        "nr_features": nr_features,
        "nr_classes": nr_classes,
        "nr_blocks": config.nr_blocks,
        "hidden_size": config.hidden_size,
        "dropout_rate": config.dropout_rate,
    }

    output_directory = os.path.join(
        config.output_dir,
        "hyconex",
        config.dataset_name,
        f"{config.seed}",
        f"{config.class_lambda}",
        f"{config.dist_lambda}",
        f"{config.flow_lambda}",
        f"{config.flow_start_epoch}",
        config.loss_type.name,
    )
    os.makedirs(output_directory, exist_ok=True)

    config.model_load_path = (
        config.model_load_path
        if config.model_load_path == ""
        else config.model_load_path
    )

    flow = MaskedAutoregressiveFlow(
        features=dataset.X_train.shape[1],
        hidden_features=16,
        context_features=1,
        num_layers=8,
        num_blocks_per_layer=4,
    )
    log_prob_threshold = 0.0

    disc_model = HyConEx(
        network_configuration,
        config=config,
        device=torch.device(config.device),
        output_directory=output_directory,
        flow=flow,
        log_prob_threshold=log_prob_threshold,
        dataset=dataset
    )

    return disc_model


def initialization() -> Tuple[HyConEx, HyConExConfig, AbstractDataset]:
    """
    Initialize Classifier model, HyConExConfig and dataset.

    Returns:
        Tuple[HyConEx, HyConExConfig, AbstractDataset]:
        Initialized model, configuration and dataset
    """

    config = config_initialization()
    set_seed(config.seed)

    wandb.init(project="HyConEx", tags=[config.dataset_name])
    wandb.config.update(OmegaConf.to_container(config))

    dataset: AbstractDataset = config.dataset.value() \
        if config.dataset != DatasetType.OPENML else config.dataset.value(config.openml_id)
    disc_model = model_initialization(dataset, config)

    if len(dataset.categorical_features) != 0:
        config.use_distance = True

    print("Numerical features: ", len(dataset.numerical_features))
    print("Categorical features: ", len(dataset.categorical_features))

    return disc_model, config, dataset
