from dataclasses import field, dataclass, fields
from enum import Enum
from typing import Type
import numpy as np
from counterfactuals.datasets import *


import torch
from torch.utils.data import DataLoader, TensorDataset


class HeartDataset:
    def __init__(self):
        data = np.load('data/heart_disease_processed.npz')
        self.X_train = data['X_train']
        self.y_train = data['y_train'].astype(np.int64)  # ← 改为int64
        self.X_val = data['X_val']
        self.y_val = data['y_val'].astype(np.int64)      # ← 改为int64
        self.X_test = data['X_test']
        self.y_test = data['y_test'].astype(np.int64)    # ← 改为int64

        # 必要属性
        self.categorical_features = []
        self.numerical_features = list(range(self.X_train.shape[1]))
        self.num_classes = 2
        self.feature_names = [f'feature_{i}' for i in range(self.X_train.shape[1])]

        # 创建PyTorch数据集 - 标签用torch.long类型
        self.train_dataset = TensorDataset(
            torch.tensor(self.X_train, dtype=torch.float32),
            torch.tensor(self.y_train, dtype=torch.long)  # ← 改为torch.long
        )
        self.val_dataset = TensorDataset(
            torch.tensor(self.X_val, dtype=torch.float32),
            torch.tensor(self.y_val, dtype=torch.long)    # ← 改为torch.long
        )
        self.test_dataset = TensorDataset(
            torch.tensor(self.X_test, dtype=torch.float32),
            torch.tensor(self.y_test, dtype=torch.long)   # ← 改为torch.long
        )

    @property
    def num_features(self):
        return self.X_train.shape[1]
    # ... 保持之前的 __init__ 和其他属性不变 ...

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

        from sklearn.cluster import KMeans
        from scipy.spatial.distance import cdist
        import numpy as np

        print("Pretrain: ", pretrain)

        # 计算数值特征的噪声水平
        self.X_noise = torch.tensor(
            [
                1 / len(np.unique(self.X_train[:, i])) * noise_factor
                for i in self.numerical_features
            ]
        )

        # 内部数据集类（复制MoonsDataset的逻辑）
        class MinDateset(torch.utils.data.Dataset):
            def __init__(self, X: np.ndarray, y: np.ndarray, y_target=None):
                self.min_targets = []

                y_t = y_target if y_target is not None else y
                unique_classes = np.unique(y_t)

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
                self.y = torch.from_numpy(y.astype(np.int64))  # ← 确保标签是整数类型
                print(X.shape[1])
                print("Targets: ", self.min_targets.shape)


            def __len__(self):
                return len(self.X)

            def __getitem__(self, idx):
                return self.X[idx], self.y[idx], self.min_targets[idx]

        # 自定义collate函数
        def collate_fn(batch):
            if not pretrain:
                X, y = zip(*batch)
            else:
                X, y, z = zip(*batch)
                z = torch.stack(z)
            X = torch.stack(X)
            y = torch.stack(y)

            # 为数值特征添加高斯噪声
            if noise_factor != 0:
                noise_level = self.X_noise
                noise = torch.randn_like(
                    X[:, self.numerical_features]) * noise_level
                X[:, self.numerical_features] = X[:,
                self.numerical_features] + noise

            # 为分类特征添加噪声（我们的数据没有分类特征，但保留逻辑）
            if cat_noise_factor != 0 and len(self.categorical_features) > 0:
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

        # 根据pretrain参数创建不同的数据集
        if not pretrain:
            dataset = TensorDataset(
                torch.from_numpy(self.X_train),
                torch.from_numpy(self.y_train)
            )
        else:
            dataset = MinDateset(self.X_train, self.y_train, y_target)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn if (noise_factor or cat_noise_factor) else None,
            **kwargs_dataloader,
        )

    # ====== 其他必要方法也需要类似的完整实现 ======
    def eval_dataloader(
            self,
            batch_size: int,
            shuffle: bool,
            *,  # 关键：* 表示后面的参数必须用关键字传递
            test,  # 这是仅关键字参数
            **kwargs_dataloader
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
    # 保持其他方法的简单实现
    def get_cv_splits(self, n_splits=5):
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        return kf.split(self.X_train)

    def get_split_data(self, split="train"):
        if split == "train":
            return self.X_train, self.y_train
        elif split in ["val", "valid", "validation"]:
            return self.X_val, self.y_val
        elif split == "test":
            return self.X_test, self.y_test
        else:
            raise ValueError(f"Unknown split: {split}")

    def load(self, file_path=None):
        return self

    def preprocess(self, **kwargs):
        return self

    def save(self, file_path):
        np.savez(
            file_path,
            X_train=self.X_train, y_train=self.y_train,
            X_val=self.X_val, y_val=self.y_val,
            X_test=self.X_test, y_test=self.y_test
        )

    def transform(self, X, y=None):
        if y is not None:
            return X, y
        return X

# ========== 添加结束 ==========


class LossType(Enum):
    CE = 0
    MULTI_DISC = 1
    DEFAULT = 2
    PRETRAIN = 3


class DatasetType(Enum):
    MOONS = MoonsDataset
    MOONS3 = Moons3Dataset
    LAW = LawDataset
    AUDIT = AuditDataset
    HELOC = HelocDataset
    DIGITS = DigitsDataset
    BLOBS = BlobsDataset
    WINE = WineDataset
    ADULT = AdultDataset
    GERMAN = GermanCreditDataset
    OPENML = OpenmlDataset
    HEART = HeartDataset


@dataclass
class HyConExConfig:
    seed: int = field(default=0, metadata={"help": "Random seed."})
    output_dir: str = field(
        default="results", metadata={"help": "Directory to save the results."}
    )
    output_csv_file: str = field(
        default="results3.csv", metadata={"help": "CSV result file path location."}
    )
    model_load_path: str = field(
        default=".", metadata={"help": "Model weights load path."}
    )
    dataset: DatasetType = field(
        default=DatasetType.MOONS, metadata={
            "help": "Counterfactual dataset type."}
    )
    dataset_name: str = field(
        default="", metadata={"help": "Alias name for dataset."}
    )
    openml_id: str = field(default="", metadata={"help": "Openml dataset id."})

    device: str = field(default="cuda:0", metadata={"help": "Device type."})

    # Model parameters
    nr_blocks: int = field(
        default=4, metadata={"help": "Number of levels in the hypernetwork."}
    )
    hidden_size: int = field(
        default=256, metadata={"help": "Number of hidden units in the hypernetwork."}
    )
    dropout_rate: float = field(
        default=0.25, metadata={"help": "Training dropout rate."}
    )
    scheduler_t_mult: int = field(
        default=2, metadata={"help": "Multiplier for the scheduler."}
    )
    nr_restarts: int = field(
        default=1, metadata={"help": "Number of learning rate restarts."}
    )

    # Training parameters
    nr_epochs: int = field(default=1500, metadata={
                           "help": "Number of train epochs."})
    batch_size: int = field(default=256, metadata={
                            "help": "Dataloader batch size."})
    learning_rate: float = field(
        default=5e-4, metadata={"help": "Learning rate value."}
    )
    cluster_lambda: float = field(
        default=0.8,
        metadata={
            "help": "Lambda for the adjustment loss term of the closest cluster in the desired class."
        },
    )
    cluster_start_epoch: int = field(
        default=100,
        metadata={
            "help": "The epoch at which the cluster adjustment loss begins to be applied."
        },
    )
    pretrain: bool = field(
        default=False, metadata={"help": "Whether to pretrain base model or not."}
    )
    pretraining_epochs: int = field(
        default=500, metadata={"help": "Number of base model pretraining epochs."}
    )
    weight_decay: float = field(default=0.01, metadata={
                                "help": "Model weight decay."})
    use_distance: bool = field(
        default=False,
        metadata={
            "help": "Whether to use distance factor during counterfactual creation."
        },
    )
    early_stopping: bool = field(
        default=True,
        metadata={
            "help": "Whether to use early stopping or not."
        },
    )

    # Training second phase parameters
    loss_type: LossType = field(
        default=LossType.CE,
        metadata={
            "help": "Type of loss function used in counterfactuals training."},
    )

    class_lambda: float = field(
        default=0.8,
        metadata={"help": "Lambda for counterfactual classification loss term."},
    )
    dist_lambda: float = field(
        default=0.1, metadata={"help": "Lambda for counterfactual distance loss term."}
    )
    flow_lambda: float = field(
        default=0.1, metadata={"help": "Lambda for counterfactual flow loss term."}
    )

    class_start_epoch: int = field(
        default=500,
        metadata={
            "help": "The epoch at which the counterfactual classification loss begins to be applied."
        },
    )
    dist_start_epoch: int = field(
        default=400,
        metadata={
            "help": "The epoch at which the counterfactual distance loss begins to be applied."
        },
    )
    flow_start_epoch: int = field(
        default=400,
        metadata={
            "help": "The epoch at which the counterfactual flow loss begins to be applied."
        },
    )

    class_warm_up_epochs: int = field(
        default=300,
        metadata={
            "help": "Number of epochs required for the counterfactual classification loss"
            "to reach its maximum lambda value."
        },
    )
    dist_warm_up_epochs: int = field(
        default=200,
        metadata={
            "help": "Number of epochs required for the counterfactual distance loss to reach its maximum lambda value."
        },
    )
    flow_warm_up_epochs: int = field(
        default=200,
        metadata={
            "help": "Number of epochs required for the counterfactual flow loss to reach its maximum lambda value."
        },
    )

    # Evaluation parameters
    full_evaluation: bool = field(
        default=False,
        metadata={"help": "Whether to evaluate all counterfactual methods."},
    )
    only_pred_eval: bool = field(
        default=False,
        metadata={"help": "Whether to evaluate quality of prediction only."},
    )
    multiple_steps: bool = field(
        default=False,
        metadata={"help": "Whether to perform multiple-step version of HyConEx explanation."},
    )
    last_step_full: bool = field(
        default=True,
        metadata={"help": "Whether to fully perform last step in multiple-step explanation."},
    )
    eps: float = field(
        default=0.05,
        metadata={"help": "Hypernetwork weights factor for multiple-step version explanation."},
    )


def print_help(cfg_cls: Type[HyConExConfig]):
    """
    Print help information from dataclass metadata.

    Parameters:
        cfg_cls (Type[HyConExConfig]): Class configuration dataclass with 'help' metadata
    """
    for f in fields(cfg_cls):
        help_text = f.metadata.get("help", "No description available.")
        print(f"{f.name}: {help_text}")


subset_columns = [
    "name",
    "coverage",
    "validity",
    "proximity_continuous_manhattan",
    "proximity_continuous_euclidean",
    "prob_plausibility",
    "log_density_cf",
    "lof_scores_cf",
    "isolation_forest_scores_cf",
    "proximity_categorical_hamming",
    "proximity_categorical_jaccard",
    "time",
]
