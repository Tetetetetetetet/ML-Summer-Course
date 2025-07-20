import json
import pandas as pd
from pathlib import Path

# 1. 路径配置
CONFIG_PATH  = Path('../config/feature.json')
TRAIN_PATH   = Path('./Data/train.csv')
TEST_PATH    = Path('./Data/test.csv')

# 2. 读取 iskeep=True 的列名
with CONFIG_PATH.open(encoding='utf-8') as f:
    cfg = json.load(f)
keep_cols = [k for k, v in cfg.get('features', {}).items() if v.get('iskeep') is True]

print(f'提取到 {len(keep_cols)} 个 iskeep=true 的特征')

# 3. 原地覆盖保存
na_vals = ["?", "None", "nan", "Unknown/Invalid",
           "NULL", "Not Available", "Not Mapped"]

# 训练集
df_train = pd.read_csv(TRAIN_PATH, na_values=na_vals)
df_train[keep_cols].to_csv(TRAIN_PATH, index=False)
print(f'已就地覆盖训练集 -> {TRAIN_PATH.resolve()}')

# 测试集
df_test = pd.read_csv(TEST_PATH, na_values=na_vals)
df_test[keep_cols].to_csv(TEST_PATH, index=False)
print(f'已就地覆盖测试集  -> {TEST_PATH.resolve()}')