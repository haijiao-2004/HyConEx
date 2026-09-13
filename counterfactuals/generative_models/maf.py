import neptune
import torch
import torch.optim as optim
import torch.nn.functional as F
from counterfactuals.generative_models import BaseGenModel
# 不再直接导入 _MaskedAutoregressiveFlow，改用标准组件构建
from nflows import transforms, distributions, flows

from tqdm import tqdm


class MaskedAutoregressiveFlow(BaseGenModel):
    def __init__(
            self,
            features,
            hidden_features,
            context_features=None,  # 保留参数但不使用，避免调用错误
            num_layers=5,
            num_blocks_per_layer=2,
            use_residual_blocks=True,
            use_random_masks=False,
            use_random_permutations=False,
            activation=F.relu,
            dropout_probability=0.0,
            batch_norm_within_layers=False,
            batch_norm_between_layers=False,
            neptune_run=None,
            device="cpu",
    ):
        super(MaskedAutoregressiveFlow, self).__init__()
        self.device = device
        self.neptune_run = neptune_run

        # ========== 关键修改开始：使用 nflows 0.14 标准方式构建无条件流 ==========
        transform_list = []
        for _ in range(num_layers):
            # 创建无条件自回归变换（不设置 context_features）
            transform_list.append(transforms.MaskedAffineAutoregressiveTransform(
                features=features,
                hidden_features=hidden_features,
                # 注意：这里不传递 context_features 参数
            ))
            transform_list.append(transforms.RandomPermutation(features=features))

        # 组合变换并创建流模型
        transform = transforms.CompositeTransform(transform_list)
        base_distribution = distributions.StandardNormal(shape=[features])
        self.model = flows.Flow(transform=transform, distribution=base_distribution)
        # ========== 关键修改结束 ==========

        # 将模型移到指定设备
        self.model.to(self.device)

    def forward(self, x, context=None):
        # ========== 关键修改：无条件流，忽略context参数 ==========
        # nflows 0.14 的无条件流 log_prob 方法不需要 context 参数
        return self.model.log_prob(inputs=x)
        # ========== 修改结束 ==========

    def fit(
            self,
            train_loader: torch.utils.data.DataLoader,
            test_loader: torch.utils.data.DataLoader,
            num_epochs: int = 100,
            learning_rate: float = 1e-3,
            patience: int = 20,
            eps: float = 1e-3,
            checkpoint_path: str = "best_model.pth",
            neptune_run: neptune.Run = None,
    ):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        patience_counter = 0
        min_test_loss = float("inf")

        for epoch in (pbar := tqdm(range(num_epochs))):
            self.train()
            train_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                labels = labels.type(torch.float32)
                optimizer.zero_grad()
                # 注意：forward调用现在只接收inputs，labels被忽略但为兼容性保留
                log_likelihood = self(inputs, labels)
                loss = -log_likelihood.mean()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            self.eval()
            test_loss = 0.0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(
                        self.device), labels.to(self.device)
                    labels = labels.type(torch.float32)
                    log_likelihood = self(inputs, labels)
                    loss = -log_likelihood.mean().item()
                    test_loss += loss
            test_loss /= len(test_loader)
            pbar.set_description(
                f"Epoch {epoch}, Train: {train_loss:.4f}, test: {test_loss:.4f}, patience: {patience_counter}"
            )
            if neptune_run:
                neptune_run["gen_train_nll"].append(train_loss)
                neptune_run["gen_test_nll"].append(test_loss)
            if test_loss < (min_test_loss + eps):
                min_test_loss = test_loss
                patience_counter = 0
                self.save(checkpoint_path)
            else:
                patience_counter += 1
            if patience_counter > patience:
                break
        self.load(checkpoint_path)

    def predict_log_prob(self, dataloader) -> torch.Tensor:
        """
        Predict log probabilities for the given dataset using the context included in the dataset.
        """
        self.eval()
        log_probs = []

        with torch.no_grad():
            for inputs, labels in dataloader:
                labels = labels.type(torch.float32)
                # 注意：labels参数仍被传递但不会被使用
                outputs = self(inputs, labels)
                log_probs.append(outputs)
        results = torch.concat(log_probs)

        assert len(dataloader.dataset) == len(results)
        return results

    # Deprecated due tu multiclass support, use self.forward instead
    # def predict_log_probs(self, X: Union[np.ndarray, torch.Tensor]):
    #     """
    #     Predict log probabilities of the input dataset for both context equal 0 and 1.
    #     Results format is of the shape: [2, N]. N is number of samples, i.e., X.shape[0].
    #     """
    #     self.eval()
    #     if isinstance(X, np.ndarray):
    #         X = torch.from_numpy(X)
    #     with torch.no_grad():
    #         y_zero = torch.zeros((X.shape[0], 1), dtype=X.dtype).to(self.device)
    #         y_one = torch.ones((X.shape[0], 1), dtype=X.dtype).to(self.device)
    #         log_p_zero = self(X, y_zero)
    #         log_p_one = self(X, y_one)
    #     result = torch.vstack([log_p_zero, log_p_one])

    #     assert result.T.shape[0] == X.shape[0], f"Shape of results don't match. " \
    #                                             f"Shape of result: {result.shape}, shape of input: {X.shape}"
    #     return result

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def _unpack_batch(self, batch):
        if isinstance(batch, tuple):
            inputs, labels = batch
            inputs, labels = inputs.to(self.device), labels.to(self.device)
        else:
            inputs, labels = batch[0], None
            inputs = inputs.to(self.device)
        return inputs, labels