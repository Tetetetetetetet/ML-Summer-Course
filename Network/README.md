# Network/net.py 使用说明

`net.py` 提供了 3 个核心函数，用于 **构建**、**训练** 和 **评估** 一个 ResNet-like 全连接网络。  
所有函数均可在其它脚本中 `from Network.net import build_model, train, evaluate` 直接调用。

---

### 1. 构建模型：`build_model`

```python
model, actual_feature_cols = build_model(feature_cols=None, n=None)
```
- feature_cols : 指定要使用的特征列编号（1-based 列表或 range）。
若为 None，则默认使用前 n 个特征（1..n）。
- n            : 当 feature_cols=None 时生效；使用前 n 个特征。

### 2. 训练：`train`

```python
train(train_csv,
      feature_cols=None,
      n=None,
      epochs=200,
      batch_size=64,
      save_path='./model/best_resnet.h5')
```
- train_csv    : 训练集 csv 路径
- feature_cols : 同 build_model
- n            : 同 build_model
- epochs       : 训练轮数
- batch_size   : 批次大小
- save_path    : 模型保存路径（含文件名）

### 3. 评估
```python
evaluate(test_csv,
         model_path,
         feature_cols=None,
         n=None)
```
- test_csv     : 测试集 csv 路径
- model_path   : 已保存的模型文件路径
- feature_cols : 同 build_model
- n            : 同 build_model

### 直接使用net.py
```linux
python -m Network.net train --n 28 --epochs 100 --save_path ./model/full.h5
python -m Network.net eval  --test_csv ./Data/test.csv --model_path ./model/full.h5 --n 28
```