import copy
import os
from pathlib import Path
from typing import Union, Self, Callable, Tuple, Dict, Any, Optional

import pandas as pd
import wandb

from counterfactuals.cf_methods.base import BaseCounterfactual, ExplanationResult
from counterfactuals.datasets.base import AbstractDataset
from counterfactuals.generative_models import MaskedAutoregressiveFlow
from hyconex import configs
from hyconex.configs import HyConExConfig, LossType
from hyconex.hypernetwork import HyperNet
from hyconex.model_utils import (
    eval_counterfactuals,
    pretraining_final_steps,
    get_classification_metrics,
    one_hot_encoder_torch, suppress_warnings
)

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR, SequentialLR
import torch
import torch.nn.functional as F
import numpy as np


def criterion_factory_multiclass(
        criterion_base: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        config: HyConExConfig,
        loss_type: LossType,
        flow: MaskedAutoregressiveFlow,
        log_prob_threshold: float,
        dataset_obj: AbstractDataset,
):
    def get_counterfact_with_mask(x: torch.Tensor, output: torch.Tensor, weights: torch.Tensor,
                                  dataset: AbstractDataset, device: torch.device):
        y_pred = torch.argmax(output, dim=-1)
        weights = weights.permute(0, 2, 1)
        w, b = weights[:, :, :-1], weights[:, :, -1:]
        B, D_out, D_in = weights.shape
        x = x.unsqueeze(1)

        # Counterfactual example creation
        distance = (torch.sum(x * w, dim=-1, keepdim=True) + b) / torch.linalg.norm(
            w, dim=-1, keepdim=True
        )
        w_unit = w / torch.linalg.norm(w, dim=-1, keepdim=True)
        x_cf = x - distance * w_unit if config.use_distance else x - w
        x_cf = x_cf.reshape(B * D_out, D_in - 1)
        x_cf = one_hot_encoder_torch(x_cf, dataset, 0.01)

        mask = torch.arange(D_out, device=device).unsqueeze(
            0) != y_pred.unsqueeze(1)
        target_values = torch.arange(
            D_out, device=device).unsqueeze(0).expand(B, D_out)
        target_values = target_values.reshape(B * D_out)

        return x_cf, x, mask, target_values, (B, D_out, D_in)

    def default(output: torch.Tensor, y: torch.Tensor, **kwargs):
        return criterion_base(output, y)

    def pretrain_loss(output: torch.Tensor, y: torch.Tensor, x: torch.Tensor, weights: torch.Tensor,
                      curr_epoch: int, hypernet: HyConEx, device: torch.device, x_target: torch.Tensor, **kwargs):
        x_cf, x, mask, target_values, shape = get_counterfact_with_mask(
            x, output, weights, dataset_obj, device
        )
        B, D_out, D_in = shape
        x_target = x_target.to(device)

        feature_scale = max(1., x.shape[2] / 50)

        cluster_loss, cluster_lambda = 0, 0
        if curr_epoch > config.cluster_start_epoch and config.cluster_lambda != 0:
            cluster_loss = (
                    torch.norm(
                        x_cf.reshape(B, D_out, D_in - 1)[
                         :, :, dataset_obj.numerical_features
                        ]
                        - x_target[:, :, dataset_obj.numerical_features],
                        dim=-1,
                        p=2.0,
                    )
                    * mask
            )

            cluster_loss += (
                    torch.norm(
                        x_cf.reshape(B, D_out, D_in - 1)[
                         :, :, dataset_obj.categorical_features
                        ]
                        - x_target[:, :, dataset_obj.categorical_features],
                        dim=-1,
                        p=2.0,
                    ) / 10
                    * mask
            )

            cluster_loss += (
                                torch.norm(
                                    x_cf.reshape(B, D_out, D_in - 1)[
                                     :, :, dataset_obj.numerical_features
                                    ]
                                    - x[:, :, dataset_obj.numerical_features],
                                    dim=-1,
                                    p=2,
                                )
                            ) * ~mask

            cluster_loss += (
                                    torch.norm(
                                        x_cf.reshape(B, D_out, D_in - 1)[
                                         :, :, dataset_obj.categorical_features
                                        ]
                                        - x[:, :, dataset_obj.categorical_features],
                                        dim=-1,
                                        p=2,
                                    ) / 10
                            ) * ~mask

            cluster_loss = cluster_loss.reshape(B * D_out)
            cluster_loss = cluster_loss.mean()

            cluster_lambda = min(
                1.0, (curr_epoch - config.cluster_start_epoch) / 200)

        cluster_lambda *= config.cluster_lambda / feature_scale

        return criterion_base(output, y) + cluster_loss * cluster_lambda

    class CounterfactualLossBase:
        def __init__(self, class_loss_func, flow: MaskedAutoregressiveFlow,
                     log_prob_threshold: float, dataset: AbstractDataset):
            self.class_loss_func = class_loss_func
            self.flow = flow
            self.log_prob_threshold = log_prob_threshold
            self.dataset = dataset

        def __call__(
                self, output: torch.Tensor, y: torch.Tensor, x: torch.Tensor, weights: torch.Tensor, curr_epoch: int,
                hypernet: HyConEx, device: torch.device, x_target: Optional[torch.Tensor], dataset_type: str
        ):
            x_cf, x, mask, target_values, shape = get_counterfact_with_mask(
                x, output, weights, self.dataset, device
            )
            B, D_out, D_in = shape

            classification_loss, class_lambda = 0, 0
            distance_loss, distance_lambda = torch.tensor(0, device=device), 0
            flow_loss, f_lambda = 0, 0
            validity, plausibility = torch.tensor(0.0), torch.tensor(0.0)
            feature_scale = max(1., x.shape[2] / 50)

            # Classification loss
            if curr_epoch > config.class_start_epoch and config.class_lambda != 0:
                output_cf_logit = hypernet(x_cf)
                output_cf_logit = output_cf_logit.reshape(
                    B, D_out, D_out)
                output_cf = torch.softmax(output_cf_logit, dim=-1)

                y_cf = torch.argmax(output_cf, dim=-1)
                validity += torch.sum(
                    (y_cf.reshape(-1) == target_values) * mask.reshape(-1)
                ).cpu() / (B * (D_out - 1))

                output_cf_logit = output_cf_logit.reshape(
                    B * D_out, D_out)

                classification_loss = self.class_loss_func(
                    output_cf=output_cf,
                    output_cf_logit=output_cf_logit,
                    target_values=target_values,
                    shape=(B, D_in, D_out),
                    device=device,
                    mask=mask,
                )

                class_lambda = min(
                    1.0,
                    (curr_epoch - config.class_start_epoch)
                    / config.class_warm_up_epochs,
                )

            # Distance loss
            if curr_epoch > config.dist_start_epoch and config.dist_lambda != 0:
                distance_loss = torch.linalg.norm(
                    x_cf.reshape(B, D_out, D_in - 1)[
                     :, :, dataset_obj.numerical_features
                    ]
                    - x[:, :, dataset_obj.numerical_features],
                    dim=-1,
                )
                cat_distance_loss = torch.linalg.norm(
                    x_cf.reshape(B, D_out, D_in - 1)[
                     :, :, dataset_obj.categorical_features
                    ]
                    - x[:, :, dataset_obj.categorical_features],
                    dim=-1,
                ) / 10
                distance_loss = distance_loss + cat_distance_loss
                distance_loss = distance_loss.reshape(B * D_out)

                distance_loss = distance_loss.mean()
                distance_lambda = min(
                    1.0,
                    (curr_epoch - config.dist_start_epoch) /
                    config.dist_warm_up_epochs,
                )

            # Flow loss
            if curr_epoch > config.flow_start_epoch and config.flow_lambda != 0:
                target_values = target_values.reshape(-1, 1)
                p_x_param_c_target: torch.Tensor = self.flow(
                    x_cf, context=target_values.type(torch.float32)
                )
                flow_loss = torch.nn.functional.relu(
                    self.log_prob_threshold - p_x_param_c_target
                ) * mask.reshape(B * D_out)

                plausibility += torch.sum(
                    (p_x_param_c_target >= self.log_prob_threshold)
                    * mask.reshape(B * D_out)
                ).cpu() / (B * (D_out - 1))

                # Compute mean only over valid values
                valid_mask = ~torch.isnan(flow_loss) & ~torch.isinf(flow_loss)
                if valid_mask.any():
                    flow_loss = (flow_loss * valid_mask).mean()
                else:
                    flow_loss = torch.tensor(0.0)

                f_lambda = min(
                    1.0,
                    (curr_epoch - config.flow_start_epoch) /
                    config.flow_warm_up_epochs,
                )

            class_lambda *= config.class_lambda
            distance_lambda *= config.dist_lambda / feature_scale
            f_lambda *= config.flow_lambda / feature_scale

            base_loss = criterion_base(output, y)

            main_loss = (
                    base_loss
                    + class_lambda * classification_loss
                    + distance_lambda * distance_loss
            )
            flow_loss_fin = f_lambda * flow_loss

            wandb.log(
                {
                    f"{dataset_type}/Base_loss": base_loss,
                    f"{dataset_type}/Classification_loss_scaled": classification_loss * class_lambda,
                    f"{dataset_type}/Distance_loss_scaled": distance_loss * distance_lambda,
                    f"{dataset_type}/Flow_loss_scaled": flow_loss * f_lambda,
                    f"{dataset_type}/epoch": curr_epoch,
                    f"{dataset_type}/validity": validity,
                    f"{dataset_type}/plausibility": plausibility,
                }
            )

            return {
                "main_loss": main_loss,
                "flow_loss": flow_loss_fin,
                "validity": validity,
                "plausibility": plausibility,
                "dist_loss": distance_loss,
            }

    def ce(
            *,
            output_cf_logit: torch.Tensor,
            target_values: torch.Tensor,
            mask: torch.Tensor,
            **kwargs,
    ):
        mask = mask.reshape(-1)
        output_cf_logit = output_cf_logit[mask]
        target_values = target_values[mask]

        classification_loss = torch.nn.functional.cross_entropy(
            output_cf_logit, target_values
        )

        return classification_loss

    def multi_disc(*, output_cf: torch.Tensor, target_values: torch.Tensor, device: torch.device,
                   shape: Tuple[int, int, int], mask: torch.Tensor, **kwargs):
        B, D_in, D_out = shape
        mask = mask.reshape(-1)
        if target_values.type() != torch.LongTensor:
            target_values = target_values.long()

        target_mask = torch.eye(
            output_cf.shape[-1], device=device)[target_values]
        target_mask = target_mask.squeeze(1)  # label 2 one-hot conversion
        non_target_mask = (~target_mask.bool()).float()
        output_cf = output_cf.reshape(B * D_out, D_out)
        p_target = torch.sum(output_cf[mask] * target_mask[mask], dim=1)
        p_max_non_target = torch.max(
            output_cf[mask] * non_target_mask[mask], dim=1
        ).values
        loss = F.relu(p_max_non_target + 0.05 - p_target)

        return loss.mean()

    if loss_type == LossType.DEFAULT:
        return default
    elif loss_type == LossType.PRETRAIN:
        return pretrain_loss
    elif loss_type == LossType.CE:
        return CounterfactualLossBase(ce, flow, log_prob_threshold, dataset_obj)
    elif loss_type == LossType.MULTI_DISC:
        return CounterfactualLossBase(multi_disc, flow, log_prob_threshold, dataset_obj)
    else:
        raise ValueError("Unexpected loss type!")


class HyConEx(torch.nn.Module, BaseCounterfactual):
    """HyConEx model class."""

    def __init__(
            self,
            network_configuration: Dict[str, Any],
            config: HyConExConfig,
            device: torch.device = "cpu",
            output_directory: str = ".",
            *,
            flow: MaskedAutoregressiveFlow,
            log_prob_threshold: float,
            dataset: AbstractDataset
    ):
        """
        Initialize the model with the given network configuration, device, and other hyperparameters.

        Parameters:
            network_configuration (Dict[str, Any]): A dictionary containing the configuration details
            for the hypernetwork.

            config (HyConExConfig): A configuration object containing additional settings specific to the model.

            device (torch.device, default="cpu"): The device on which to run the computations.

            output_directory (str, default="."): The directory where output files will be saved.

            flow (MaskedAutoregressiveFlow): The flow model to be used for density estimation.

            log_prob_threshold (float): The threshold for log probability to be used for plausibility estimation.

            dataset (AbstractDataset): The dataset to be used for training or inference.
        """
        super(HyConEx, self).__init__()

        self.nr_classes: int = network_configuration["nr_classes"]

        self.model = HyperNet(**network_configuration)
        self.model = self.model.to(device)
        self.flow = flow
        self.log_prob_threshold = log_prob_threshold

        self.config = config
        self.dev = device

        self.dataset = dataset

        self.softmax_act_func = torch.nn.Softmax(dim=1)
        self.output_directory = output_directory
        print("Output path", self.output_directory)

        self.patience = 200
        self.best_validity, self.best_plausibility, self.best_loss = 0.0, 0.0, 0.0
        self.best_state = None

    def _loop_step(self, x: torch.Tensor, y: torch.Tensor, x_target: Optional[torch.Tensor], criterion: Callable,
                   epoch: int, auroc: torch.Tensor, acc: torch.Tensor, loss_value: float, validity: float,
                   plausibility: float, dist_loss: float, dataset_type: str
                   ):
        x, y = x.to(self.dev), y.to(self.dev)
        batch_size = len(x)

        if dataset_type == "Val":
            with torch.no_grad():
                output, weights = self.model(
                    x, return_weights=True, simple_weights=True)
        else:
            output, weights = self.model(
                x, return_weights=True, simple_weights=True)
        loss = criterion(output, y, x, weights, epoch,
                         self.model, self.dev, x_target, dataset_type=dataset_type)
        if isinstance(loss, dict):
            flow_loss = loss["flow_loss"]
            batch_loss = loss["main_loss"] + flow_loss
            validity += batch_size * loss["validity"].item()
            plausibility += batch_size * loss["plausibility"].item()
            dist_loss += batch_size * loss["dist_loss"].item()
        else:
            batch_loss = loss

        predictions = torch.softmax(output, dim=1)
        batch_auroc, batch_acc = get_classification_metrics(
            predictions, y, self.nr_classes
        )

        loss_value += batch_size * batch_loss.item()
        auroc += batch_size * batch_auroc
        acc += batch_size * batch_acc

        return loss, auroc, acc, loss_value, validity, plausibility, dist_loss

    def _log_epoch_metrics(self, loss: float, flow_loss: float, dist_loss: float, auroc: float, acc: float,
                           validity: float, plausibility: float, dataset_size: int, epoch: int, dataset_type: str
                           ):
        loss /= dataset_size
        flow_loss /= dataset_size
        auroc /= dataset_size
        acc /= dataset_size
        validity /= dataset_size
        plausibility /= dataset_size
        dist_loss /= dataset_size

        print(
            f"{dataset_type} Epoch: {epoch}, Loss: {loss}, flow_loss: {flow_loss}, "
            f"AUROC: {auroc}, Accuracy: {acc}, Validity: {validity} Plausibility: {plausibility}"
        )
        wandb.log(
            {
                f"{dataset_type}/loss_epoch": loss,
                f"{dataset_type}/auroc_epoch": auroc,
                f"{dataset_type}/accuracy_epoch": acc,
                f"{dataset_type}/validity_epoch": validity,
                f"{dataset_type}/plausibility_epoch": plausibility,
            }
        )

        return acc, validity, plausibility, dist_loss

    def _save_model(self, path: str):
        checkpoint = {
            "model": self.model.state_dict(),
            "flow": self.flow.state_dict(),
            "log_prob_threshold": self.log_prob_threshold,
        }
        torch.save(checkpoint, path)

    def _update_best_val_params(self, val_validity: float, val_plausibility: float, val_dist_loss: float):
        self.best_validity = val_validity
        self.best_plausibility = max(self.best_plausibility, val_plausibility)
        self.best_loss = val_dist_loss
        self.best_state = copy.deepcopy(self.model.state_dict())
        self.patience = 200
        print(
            f"New best validation epoch, validity: {self.best_validity} "
            f"plausibility: {self.best_plausibility} "
            f"distl_loss {self.best_loss}"
        )

    def _load_from_checkpoint(self, load_path: str, optimizer: torch.optim.Optimizer,
                              scheduler: torch.optim.lr_scheduler.SequentialLR, criterion):
        checkpoint = torch.load(load_path, weights_only=False)

        self.model.load_state_dict(checkpoint["model"])

        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

        log_prob_threshold = checkpoint["log_prob_threshold"]
        criterion.log_prob_threshold = log_prob_threshold
        self.log_prob_threshold = log_prob_threshold
        print("Loaded log prob threshold ", log_prob_threshold)

        criterion.flow.load_state_dict(checkpoint["flow"])
        self.flow = criterion.flow
        criterion.flow.to(self.dev)

    def _phase_fit(
            self,
            train_loader: torch.utils.data.DataLoader,
            pretrain: bool = False,
    ):
        self.to(self.config.device)
        # Optimizer and scheduler setup
        T_0: int = max(
            (
                    (self.config.nr_epochs * len(train_loader))
                    * (self.config.scheduler_t_mult - 1)
            )
            // (self.config.scheduler_t_mult ** self.config.nr_restarts - 1),
            1,
        )
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler2 = CosineAnnealingWarmRestarts(
            optimizer, 2 * T_0, self.config.scheduler_t_mult
        )

        def warmup(current_step: int):
            return float(current_step / (5 * len(train_loader)))

        scheduler1 = LambdaLR(optimizer, lr_lambda=warmup)
        scheduler = SequentialLR(
            optimizer,
            schedulers=[scheduler1, scheduler2],
            milestones=[5 * len(train_loader)],
        )

        criterion_base = torch.nn.CrossEntropyLoss()
        criterion = criterion_factory_multiclass(
            criterion_base,
            self.config,
            LossType.PRETRAIN if pretrain else self.config.loss_type,
            self.flow,
            self.log_prob_threshold,
            self.dataset,
        )

        start_epoch = 1
        base_acc, base_auroc = 0., 0.
        # Load checkpoint if pretrained
        if not pretrain and self.config.model_load_path != "":
            start_epoch = min(
                self.config.dist_start_epoch,
                self.config.class_start_epoch,
                self.config.flow_start_epoch,
            )
            checkpoint = torch.load(os.path.join(self.config.model_load_path, "base.pt"))
            self.model.load_state_dict(checkpoint["model"])

            eval_predictions = self.predict_proba(torch.tensor(self.dataset.X_val)).to(self.dev)
            base_auroc, base_acc = get_classification_metrics(
                eval_predictions, torch.from_numpy(self.dataset.y_val).to(self.dev), self.nr_classes
            )
            print("Base model val acc: ", base_acc, " Base model val auroc: ", base_auroc)

            self._load_from_checkpoint(os.path.join(self.config.model_load_path, "checkpoint.pt"),
                                       optimizer, scheduler, criterion)

        eval_dataloader = self.dataset.eval_dataloader(
            self.config.batch_size, shuffle=False, test=False
        )

        self.eval()
        output = self.model(torch.tensor(self.dataset.X_val, device=self.dev))
        predictions = torch.softmax(output, dim=-1)
        y_val_true = torch.tensor(self.dataset.y_val, device=self.dev)

        checkpoint_auroc, checkpoint_acc = get_classification_metrics(
            predictions, y_val_true, self.nr_classes
        )
        start_acc = max(base_acc, checkpoint_acc.item())
        start_auroc = max(base_auroc, checkpoint_auroc.item())

        print("Start val acc: ", start_acc)
        print("Start val aucroc: ", start_auroc)

        self.patience = 200
        self.best_validity, self.best_plausibility, self.best_loss = 0.0, 0.0, 0.0
        self.best_state = None
        epoch, last_epoch = 0, 0

        for epoch in range(start_epoch, self.config.nr_epochs + 1):
            self.train()

            # Update train loader with model predicted y_target values
            if pretrain and self.config.cluster_start_epoch == epoch:
                y_pred_train = self.predict(
                    torch.tensor(self.dataset.X_train).to(self.dev))
                y_pred_train = y_pred_train.long().detach().numpy()
                train_loader = self.dataset.train_dataloader(
                    batch_size=self.config.batch_size,
                    shuffle=True,
                    pretrain=True,
                    y_target=y_pred_train,
                )
                self._save_model(os.path.join(self.output_directory, "base.pt"))

            # Training step
            train_loss, train_flow_loss, train_validity, train_plausibility = 0, 0, 0.0, 0.0
            train_auroc, train_acc = torch.tensor(0.0, device=self.dev), torch.tensor(0.0, device=self.dev)

            for batch_idx, batch in enumerate(train_loader):
                self.model.train()
                optimizer.zero_grad()

                if not pretrain:
                    x, y = batch
                    x_target = None
                else:
                    x, y, x_target = batch

                train_step_values = self._loop_step(x, y, x_target, criterion, epoch, train_auroc, train_acc,
                                                    train_loss, train_validity, train_plausibility,
                                                    0.0, "Train")
                loss, train_auroc, train_acc, train_loss, train_validity, train_plausibility, _ = train_step_values

                if isinstance(loss, dict):
                    flow_loss = loss["flow_loss"]
                    main_loss = loss["main_loss"]

                    clip_value = 1.0
                    if isinstance(flow_loss, torch.Tensor):
                        if not flow_loss.isnan() and not flow_loss.isinf():
                            flow_loss.backward(retain_graph=True)
                            torch.nn.utils.clip_grad_value_(
                                self.model.parameters(), clip_value=clip_value
                            )
                        else:
                            flow_loss = 0
                            print(
                                "NaN/INF", loss["main_loss"].item(), flow_loss)

                    train_flow_loss += len(x) * flow_loss
                    main_loss.backward()
                else:
                    loss.backward()

                optimizer.step()
                scheduler.step()

            dataset_size = len(self.dataset.X_train)
            self._log_epoch_metrics(
                train_loss,
                train_flow_loss,
                0.0,
                train_auroc.item(),
                train_acc.item(),
                train_validity,
                train_plausibility,
                dataset_size,
                epoch,
                "Train",
            )

            if not pretrain:
                # Validation step
                val_loss, val_flow_loss, val_validity, val_plausibility, val_dist_loss = 0, 0, 0.0, 0.0, 0.0
                val_auroc, val_acc = torch.tensor(0.0, device=self.dev), torch.tensor(0.0, device=self.dev)

                for batch_idx, batch in enumerate(eval_dataloader):
                    x, y = batch
                    self.model.eval()

                    val_step_values = self._loop_step(x, y, None, criterion, epoch, val_auroc, val_acc,
                                                      val_loss, val_validity, val_plausibility,
                                                      val_dist_loss, "Val")
                    loss, val_auroc, val_acc, val_loss, val_validity, val_plausibility, val_dist_loss = val_step_values

                    if isinstance(loss, dict):
                        flow_loss = loss["flow_loss"]
                        if isinstance(flow_loss, torch.Tensor):
                            if flow_loss.isnan() or flow_loss.isinf():
                                flow_loss = 0
                                print(
                                    "NaN/INF", loss["main_loss"].item(), flow_loss)

                        val_flow_loss += len(x) * flow_loss
                dataset_size = len(self.dataset.X_val)
                epoch_acc, epoch_validity, epoch_plausibility, epoch_dist_loss = (
                    self._log_epoch_metrics(
                        val_loss,
                        val_flow_loss,
                        val_dist_loss,
                        val_auroc.item(),
                        val_acc.item(),
                        val_validity,
                        val_plausibility,
                        dataset_size,
                        epoch,
                        "Val",
                    )
                )

                start_early_exit = max(
                    self.config.dist_start_epoch,
                    self.config.class_start_epoch,
                    self.config.flow_start_epoch,
                )
                self.best_state = (
                    copy.deepcopy(self.model.state_dict())
                    if epoch == start_early_exit
                    else self.best_state
                )
                self.patience = 200 if epoch == start_early_exit else self.patience

                self.patience -= 1
                if epoch >= start_early_exit:
                    if epoch_acc - start_acc >= -0.03:
                        if self.best_validity < epoch_validity < 0.95:
                            self._update_best_val_params(
                                epoch_validity, epoch_plausibility, epoch_dist_loss
                            )
                        elif epoch_validity > 0.95:
                            if self.best_plausibility < epoch_plausibility or self.best_validity < 0.95:
                                self._update_best_val_params(
                                    epoch_validity, epoch_plausibility, epoch_dist_loss
                                )
                            else:
                                if (
                                        self.best_loss > epoch_dist_loss
                                        and epoch_plausibility
                                        >= 0.95 * self.best_plausibility
                                ):
                                    self._update_best_val_params(
                                        epoch_validity,
                                        epoch_plausibility,
                                        epoch_dist_loss,
                                    )
            if self.patience < 0 and self.best_validity > 0.99 and self.config.early_stopping:
                break

            if self.patience == 200 and epoch - last_epoch > 50:
                last_epoch = epoch
                df, auroc, acc = eval_counterfactuals(
                    self, self.dataset, self.config.use_distance, y_val_true
                )

                df = df[configs.subset_columns]
                df["epoch"] = epoch
                df = df.groupby("epoch").mean(numeric_only=True).reset_index()

                df["dataset"] = self.config.dataset_name
                df["epoch"] = epoch
                df["dist-lambda"] = self.config.dist_lambda
                df["proj-lambda"] = self.config.class_lambda
                df["flow-lambda"] = self.config.flow_lambda
                df["train-auroc"] = train_auroc.item()
                df["train-acc"] = train_acc.item()
                df["test-auroc"] = auroc.item()
                df["test-acc"] = acc.item()

                if not os.path.exists(self.config.output_csv_file):
                    path = Path(self.config.output_csv_file)
                    path.touch()

                df_saved = (
                    pd.read_csv(self.config.output_csv_file)
                    if os.path.getsize(self.config.output_csv_file) > 0
                    else pd.DataFrame()
                )
                df_combined = pd.concat([df_saved, df], ignore_index=True)
                df_combined.to_csv(
                    self.config.output_csv_file, index=False, float_format="%.4f"
                )

            if epoch == self.config.pretraining_epochs and pretrain:
                return pretraining_final_steps(
                    self, self.dataset, optimizer, scheduler
                )

        if self.best_state is not None and self.config.early_stopping:
            self.model.load_state_dict(self.best_state)
        self._save_model(os.path.join(
            self.output_directory, f"best_model_{epoch}.pt"))
        self._save_model(os.path.join(self.output_directory, f"best_model.pt"))
        print(
            f"Best validation, validity: {self.best_validity} "
            f"plausibility: {self.best_plausibility}"
            f"distl_loss {self.best_loss}"
        )

        return self

    def _pred(
            self,
            X_test: Union[torch.Tensor, np.ndarray],
            y_test: Optional[Union[torch.Tensor, np.ndarray]] = None,
            return_weights: bool = False,
            return_only_weights: bool = False,
            return_all_weights: bool = False,
            simple_weights: bool = False,
    ):
        X_test = torch.tensor(X_test).float()
        X_test = X_test.to(self.dev)

        if y_test is not None:
            y_test = torch.tensor(y_test).long()
        else:
            y_test = torch.zeros(X_test.size(0)).long()
        y_test = y_test.to(self.dev)

        predictions = []
        weights = []

        for snapshot_idx, snapshot in enumerate([self.model.state_dict()]):
            self.model.load_state_dict(snapshot)
            self.model.eval()
            if return_weights:
                output, model_weights = self.model(
                    X_test, return_weights=True, simple_weights=simple_weights
                )
                weights.append(model_weights.detach())
            else:
                output, model_weights = self.model(
                    X_test, return_weights=True, simple_weights=simple_weights
                )

            output = output.squeeze(1)
            output = self.softmax_act_func(output)

            predictions.append(output.detach())

        predictions = torch.stack(predictions, dim=0)
        predictions = torch.mean(predictions, dim=0)
        predictions = torch.squeeze(predictions)

        if return_only_weights:
            return torch.squeeze(weights[-1])

        if return_weights:
            weights = weights[-1]
            weights = torch.squeeze(weights)
            if not return_all_weights:
                act_predictions = torch.argmax(predictions, dim=1)

                correct_predictions_mask = act_predictions == y_test
                weights = weights[correct_predictions_mask]

        if return_weights:
            return predictions, weights
        else:
            return predictions

    def fit(self) -> Self:
        """
        Train the model with the provided data during initialization.

        Returns:
            Self: The fitted model instance.
        """

        y_train_true = self.dataset.y_train
        y_val_true = self.dataset.y_val

        if self.config.pretrain:
            train_dataloader = self.dataset.train_dataloader(
                batch_size=self.config.batch_size,
                shuffle=True,
                pretrain=True,
            )
            self._phase_fit(
                train_dataloader,
                pretrain=True,
            )
            self.config.model_load_path = self.output_directory

        self.dataset.y_train = y_train_true
        self.dataset.y_val = y_val_true
        train_dataloader = self.dataset.train_dataloader(batch_size=self.config.batch_size, shuffle=True)
        self._phase_fit(train_dataloader, pretrain=False)

        return self

    def predict_proba(self, X_test: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """
        Predict the probability estimates for each class for the given input data.

        Parameters:
            X_test (Union[torch.Tensor, np.ndarray]): The input data for which to predict probabilities.

        Returns:
            Union[torch.Tensor, np.ndarray]: The predicted probabilities for each class.
        """

        probs = self._pred(X_test)
        if X_test.shape[0] == 1:
            probs = probs.unsqueeze(0)
        probs = probs.detach().cpu()

        return probs if type(X_test) == torch.Tensor else probs.numpy()

    def predict(self, X_test: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """
        Predict the class labels for the given input data.

        Parameters:
            X_test (Union[torch.Tensor, np.ndarray]): The input data for which to predict the class labels.

        Returns:
            Union[torch.Tensor, np.ndarray]: The predicted class labels for each sample in `X_test`.
        """

        X_torch = torch.tensor(X_test)
        with torch.no_grad():
            probs = self.predict_proba(X_torch)
        preds = torch.argmax(probs, dim=-1).squeeze().float()

        return preds if type(X_test) == torch.Tensor else preds.numpy()

    def to(self, device):
        self.model = self.model.to(device)
        self.flow = self.flow.to(device)
        self.dev = device

        return self

    def eval(self):
        self.model.eval()

        return self

    def train(self, mode=True):
        self.model.train(mode)

        return self

    def parameters(self, recurse=True):
        return self.model.parameters(recurse)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.model(X)

    def explain(self, X: np.ndarray, y_origin: np.ndarray, y_target: np.ndarray, X_train: np.ndarray,
                y_train: np.ndarray, **kwargs) -> ExplanationResult:
        raise NotImplementedError(
            "This method is not implemented for this class.")

    def _get_counterfactual(self, x: torch.Tensor, target_class_value: Optional[int], contexts_origin: torch.Tensor,
                            use_distance: bool, eps: float = 1.0, xs_origin: Optional[torch.Tensor] = None):
        output, weights = self.model(
            x, return_weights=True, simple_weights=True
        )

        if target_class_value is None:
            contexts_target = (1 - contexts_origin).long()
            weights = weights[
                      torch.arange(weights.shape[0]), :, contexts_target.squeeze(-1)
                      ]
        else:
            weights = weights[:, :, target_class_value]
            contexts_target = target_class_value * torch.ones(x.shape[0], 1)

        weights = torch.squeeze(weights, dim=-1)
        w, b = weights[:, :-1], weights[:, -1:]
        w_unit = w / torch.linalg.norm(w, dim=-1, keepdim=True)

        xs_origin = xs_origin if xs_origin is not None else x
        distance = (
                           torch.sum(xs_origin * w, dim=-1, keepdim=True) + b
                   ) / torch.linalg.norm(w, dim=-1, keepdim=True)

        w = distance * w_unit if use_distance else w
        x_cf = x - eps * w

        return x_cf, contexts_target, output

    @suppress_warnings
    def explain_dataloader(self, dataloader: torch.utils.data.DataLoader, target_class_value: Optional[int] = None,
                           multiple_steps: bool = False, last_step_full: bool = False, eps: float = 0.05,
                           steps: int = 1000, *, use_distance: bool, **search_step_kwargs
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Search counterfactual explanations for the given dataloader.

        Parameters:
            dataloader (torch.utils.data.DataLoader): A PyTorch DataLoader containing the dataset
                                                       for which explanations are to be generated.

            target_class_value (Optional[int], default=None): The target class value to focus on
            when explaining predictions. If `None`, explanations are provided for all classes.
            If `self.nr_classes > 2`, this argument **must** be specified to avoid errors.

            multiple_steps (bool, default=False): Whether to perform multiple-step version of HyConEx explanation.

            last_step_full (bool, default=False): Whether to fully perform last step in multiple-step explanation.

            eps (float, default=0.05): Hypernetwork weights factor for multiple-step version explanation.

            steps (int, default=1000): Maximum number of steps for multiple-step version explanation.

            use_distance (bool): Whether to use distance factor during counterfactual creation.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            A tuple containing five numpy arrays:
                - Explanation values for each sample (first array).
                - Additional explanation-related components (second to fifth arrays).

        Raises:
        ValueError: If `self.nr_classes > 2` and `target_class_value` is `None`.
                    In multi-class classification, `target_class_value` must be specified.
        """
        if self.nr_classes > 2 and target_class_value is None:
            raise ValueError("When the number of classes is greater than 2, 'target_class_value' must be specified.")

        self.model.eval()

        target_class = []
        original = []
        original_class = []
        counterfactual = []

        for xs_origin, contexts_origin in dataloader:
            xs_origin = xs_origin.to(self.dev)
            contexts_origin = contexts_origin.to(self.dev)
            contexts_origin = contexts_origin.reshape(-1, 1)

            if not multiple_steps:
                x_cf, contexts_target, _ = self._get_counterfactual(xs_origin, target_class_value,
                                                                    contexts_origin, use_distance)
            else:
                if target_class_value is None:
                    contexts_target = 1 - contexts_origin
                else:
                    contexts_target = target_class_value * torch.ones(xs_origin.shape[0], 1, device=self.dev)

                batch_proj = [xs_origin.unsqueeze(0).detach().cpu()]
                batch_idx = torch.zeros_like(contexts_origin)
                border_crossed = torch.zeros_like(contexts_origin)

                x_cf = xs_origin
                for i in range(steps):
                    x_cf, _, output = self._get_counterfactual(x_cf, target_class_value, contexts_origin, use_distance,
                                                               eps=eps, xs_origin=xs_origin)
                    x_cf = one_hot_encoder_torch(x_cf, self.dataset, 0.01)

                    batch_proj.append(x_cf.unsqueeze(0).detach().cpu())

                    y_pred = torch.argmax(output, dim=-1, keepdim=True)
                    border_crossed = torch.minimum(border_crossed + (y_pred == contexts_target).float(),
                                                   torch.tensor(1))
                    batch_idx += (y_pred != contexts_target).float() * (1 - border_crossed)

                    if i % 25 == 0 and torch.all(border_crossed == torch.tensor(1)):
                        break

                if not last_step_full:
                    batch_idx = batch_idx.int().squeeze(-1)
                    batch_proj = torch.vstack(batch_proj).permute(1, 0, 2)
                    x_cf = batch_proj[torch.arange(xs_origin.shape[0]), batch_idx.to('cpu')].to(self.dev)
                else:
                    batch_idx = batch_idx.int().squeeze(-1) - 1
                    batch_proj = torch.vstack(batch_proj).permute(1, 0, 2)
                    x_cf = batch_proj[torch.arange(xs_origin.shape[0]), batch_idx.to('cpu')].to(self.dev)

                    x_cf, _, _ = self._get_counterfactual(x_cf, target_class_value,
                                                          contexts_origin, use_distance)

            counterfactual.append(x_cf.detach().cpu().numpy())
            original.append(xs_origin.detach().cpu().numpy())
            original_class.append(contexts_origin.detach().cpu().numpy())
            target_class.append(contexts_target.detach().cpu().numpy())

        X_cf = np.concatenate(counterfactual, axis=0)
        return (
            X_cf,
            np.concatenate(original, axis=0),
            np.concatenate(original_class, axis=0),
            np.concatenate(target_class, axis=0),
            np.ones(X_cf.shape[0]),
        )