import sys

sys.path.append('.')
import numpy as np
import torch

print("验证 HeartDataset 完整性...")

try:
    from hyconex.configs import HeartDataset

    # 1. 实例化
    dataset = HeartDataset()
    print("✓ 1. 实例化成功")
    print(f"   - 特征数: {dataset.num_features}")
    print(f"   - 训练样本: {len(dataset.X_train)}")
    print(f"   - 分类特征: {dataset.categorical_features}")
    print(f"   - 数值特征: {dataset.numerical_features}")

    # 2. 测试 train_dataloader 的各种调用方式
    print("\n✓ 2. 测试 train_dataloader:")

    # 预训练模式
    loader1 = dataset.train_dataloader(batch_size=32, shuffle=True, pretrain=True)
    print(f"   - pretrain=True: 通过，返回类型: {type(loader1)}")

    # 非预训练模式
    loader2 = dataset.train_dataloader(batch_size=32, shuffle=True, pretrain=False)
    print(f"   - pretrain=False: 通过，返回类型: {type(loader2)}")

    # 带噪声参数
    loader3 = dataset.train_dataloader(batch_size=32, shuffle=True, noise_factor=0.1)
    print(f"   - 带noise_factor: 通过")

    # 3. 测试 eval_dataloader
    print("\n✓ 3. 测试 eval_dataloader:")
    eval_loader = dataset.eval_dataloader(batch_size=32, shuffle=False)
    print(f"   - eval_dataloader: 通过")

    # 4. 测试其他必要方法
    print("\n✓ 4. 测试其他方法:")
    print(f"   - get_cv_splits: {dataset.get_cv_splits(3) is not None}")
    X_train, y_train = dataset.get_split_data("train")
    print(f"   - get_split_data('train'): X.shape={X_train.shape}, y.shape={y_train.shape}")

    print("\n" + "=" * 50)
    print("✅ 所有验证通过！可以开始训练了！")
    print("=" * 50)

except Exception as e:
    print(f"\n❌ 验证失败: {e}")
    import traceback

    traceback.print_exc()