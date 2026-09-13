"""
简化版：仅准备心脏病数据集供HyConEx使用
"""
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

print("=" * 60)
print("开始准备心脏病数据集")
print("=" * 60)

# 1. 从OpenML加载数据
print("1. 从Scikit-learn加载心脏病数据集...")
try:
    heart_bunch = fetch_openml(name='heart-disease', version=1, as_frame=True, parser='pandas')
    df = heart_bunch.frame
    print(f"   ✓ 加载成功，数据集形状: {df.shape}")
    print(f"   ✓ 特征列: {list(df.columns)}")
except Exception as e:
    print(f"   ✗ 加载失败: {e}")
    print("   尝试备用方案...")
    # 如果在线加载失败，这里可以添加从本地CSV加载的代码
    sys.exit(1)

# 2. 准备目标变量 (二分类：是否有心脏病)
print("\n2. 准备目标变量...")
# 查找目标列 - 常见列名
possible_targets = ['target', 'class', 'disease', 'num', 'result']
target_col = None

for col in possible_targets:
    if col in df.columns:
        target_col = col
        break

if target_col is None:
    # 如果没找到，假设最后一列是目标
    target_col = df.columns[-1]
    print(f"   警告：未找到标准目标列，使用最后一列 '{target_col}'")

print(f"   ✓ 目标列: '{target_col}'")
print(f"   ✓ 原始目标值分布:")
print(df[target_col].value_counts().sort_index())

# 转换为二分类：0=健康，1=患病
y_raw = df[target_col].astype(int).values
y = (y_raw > 0).astype(np.float32)  # 0保持为0，>0变为1

print(f"   ✓ 二值化后 - 健康(0): {(y == 0).sum()}, 患病(1): {(y == 1).sum()}")

# 3. 准备特征
print("\n3. 准备特征数据...")
X = df.drop(columns=[target_col])

# 处理非数值特征：自动识别并独热编码
categorical_cols = X.select_dtypes(include=['object', 'category']).columns
if len(categorical_cols) > 0:
    print(f"   检测到分类特征: {list(categorical_cols)}")
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    print(f"   独热编码后特征数: {X.shape[1]}")
else:
    print("   未检测到分类特征，使用原始数值特征")

# 确保所有数据为float32
X = X.astype(np.float32)

# 4. 数据集划分 (训练60%，验证20%，测试20%)
print("\n4. 划分数据集...")
X_temp, X_test, y_temp, y_test = train_test_split(
    X.values, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp  # 0.25 * 0.8 = 0.2
)

print(f"   ✓ 训练集: {X_train.shape[0]} 样本")
print(f"   ✓ 验证集: {X_val.shape[0]} 样本")
print(f"   ✓ 测试集: {X_test.shape[0]} 样本")
print(f"   ✓ 特征数: {X_train.shape[1]}")

# 5. 特征标准化
print("\n5. 标准化特征...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
print("   ✓ 标准化完成")

# 6. 保存数据
print("\n6. 保存数据集文件...")
os.makedirs('data', exist_ok=True)
save_path = 'data/heart_disease_processed.npz'

np.savez(save_path,
         X_train=X_train.astype(np.float32),
         y_train=y_train.astype(np.float32),
         X_val=X_val.astype(np.float32),
         y_val=y_val.astype(np.float32),
         X_test=X_test.astype(np.float32),
         y_test=y_test.astype(np.float32))

print(f"   ✓ 数据已保存至: {save_path}")

# 7. 显示下一步操作指南
print("\n" + "=" * 60)
print("数据准备完成！接下来需要：")
print("=" * 60)
print("\n【第一步】手动修改 datasets.py 文件")
print("-" * 40)
print("找到项目中的 datasets.py 文件（通常在 hyconex/ 目录下）")
print("添加以下代码：")
print("""
class HeartDataset:
    def __init__(self):
        data = np.load('data/heart_disease_processed.npz')
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']
        self.X_test = data['X_test']
        self.y_test = data['y_test']

    @property
    def num_features(self):
        return self.X_train.shape[1]
""")
print("\n并在 DATASETS 字典中添加：")
print("    'HEART': HeartDataset,")

print("\n【第二步】运行训练命令")
print("-" * 40)
print("1. 预训练：")
print("""
python train_model.py dataset=HEART pretrain=True \\
    class_lambda=0.9 dist_lambda=0.05 flow_lambda=0.05 \\
    num_epochs=300
""")

print("\n2. 主训练（需替换路径）：")
print("""
python train_model.py dataset=HEART pretrain=False \\
    model_load_path="results/hyconex/HEART/.../checkpoint.pt" \\
    class_lambda=0.8 dist_lambda=0.1 flow_lambda=0.1 \\
    num_epochs=500
""")

print("\n提示：model_load_path 需要替换为第一步预训练后生成的检查点实际路径")
print("\n" + "=" * 60)