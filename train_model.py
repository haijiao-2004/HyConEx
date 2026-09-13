import torch.serialization
from omegaconf import DictConfig
from omegaconf.base import ContainerMetadata

# 将 omegaconf 的相关类加入安全加载的白名单
torch.serialization.add_safe_globals([DictConfig, ContainerMetadata])
# 将 omegaconf.DictConfig 加入安全加载的白名单
torch.serialization.add_safe_globals([DictConfig])

import torch
import wandb

from hyconex.model_utils import eval_counterfactuals
from hyconex.initialization import initialization


def train():
    disc_model, config, dataset = initialization()

    y_val_true = dataset.y_val
    disc_model.fit()

    eval_counterfactuals(
        disc_model,
        dataset,
        config.use_distance,
        torch.tensor(y_val_true).to(config.device),
    )

    wandb.finish()

    return disc_model, config, dataset


if __name__ == "__main__":
    train()
