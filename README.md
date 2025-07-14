# 糖尿病数据集分析项目

## 项目概述

本项目对糖尿病数据集进行全面的数据分析和机器学习建模，包括数据预处理、缺失值填充、特征工程和模型训练。

## 环境配置

### 快速安装（推荐）(linux/macos)

```bash
./setup.sh
```

### 手动安装
#### 通过 environment.yml

```bash
# 1. 创建环境
conda env create -f environment.yml
conda activate 309

# 2. 安装自定义工具包
cd myutils
pip install -e .
```

#### 通过 requirements.txt
```
conda create -n ml python=3.9
conda install pip
pip install -r requirements.txt
cd myutils && pip install -e .
```

## 运行

### for linux/macos

#### 完整pipeline
```bash
make
```

#### 分步运行

**1. 数据预处理**
```bash
make process
```

**2. 改进的逻辑回归缺失值填充**
```bash
make impute
```

**3. 模型训练**
```bash
# 不使用过采样
make train

# 使用过采样
make train OVERSAMPLE=true
```

### 预期结果

**数据预处理结果：**
```
[INFO] 训练集: (90105, 50) -> (90105, 50)
[INFO] 测试集: (10009, 50) -> (10009, 50)
```

**改进的逻辑回归填值结果：**
```
[INFO] ==========改进的逻辑回归填补流程开始==========
[INFO] 训练集缺失值: 350852
[INFO] 测试集缺失值: 38842
[INFO] 填补后训练集缺失值: 0
[INFO] 填补后测试集缺失值: 0
[INFO] ==========改进的逻辑回归填补流程完成==========
```

**模型训练结果：**
```
[INFO] ==========建模总结==========
[INFO] 数据源: improved_logistic_imputed
[INFO] 最佳模型: RandomForest
[INFO] 最佳准确率: 0.5846
[INFO] ==========建模流程完成==========
```

### for windows

**处理数据**
```bash
python src/data_process.py
python src/data_process_test.py
python src/improved_logistic_imputation.py
```

**训练模型**
```bash
python src/data_fit.py
```

## 结果文件

### 数据预处理结果
- `Dataset/processed/train_processed/recoded_train.csv`: 重新编码后的训练数据集，缺失值记为None
- `Dataset/processed/test_processed/recoded_test.csv`: 重新编码后的测试数据集，缺失值记为None

### 改进的逻辑回归填值结果
- `Dataset/processed/train_processed/improved_logistic_imputed/improved_logistic_imputed_train_final.csv`: 改进逻辑回归填值后的训练数据集，可直接用于训练
- `Dataset/processed/train_processed/improved_logistic_imputed/improved_logistic_imputed_test_final.csv`: 改进逻辑回归填值后的测试数据集，可直接用于测试
- `Dataset/processed/train_processed/improved_logistic_imputed/imputation_report.json`: 填值报告
- `Dataset/processed/train_processed/improved_logistic_imputed/model_info.json`: 模型信息

### 模型训练结果
- `Dataset/processed/train_processed/modeling_results/modeling_report.json`: 建模报告
- `Dataset/processed/train_processed/modeling_results/best_model_*.pkl`: 最佳模型文件
- `Dataset/processed/train_processed/modeling_results/predictions_*.csv`: 预测结果

## 改进的逻辑回归填值方法

### 主要改进
1. **数据标准化**: 使用StandardScaler对数值特征进行标准化，解决收敛问题
2. **多solver尝试**: 自动尝试不同的优化算法（lbfgs, liblinear, saga）
3. **鲁棒性增强**: 当模型训练失败时，使用备选填充方法
4. **错误处理**: 完善的异常处理机制，确保填值过程不会中断
5. **测试集支持**: 使用训练好的模型对测试集进行一致的填值

### 解决的核心问题
- **迭代限制问题**: 通过数据标准化和多solver尝试解决"STOP: TOTAL NO. of ITERATIONS REACHED LIMIT"
- **数值稳定性**: 处理矩阵运算中的除零、溢出等问题
- **特征兼容性**: 确保训练集和测试集的特征处理一致

## 如何修改

### 尝试新的数据预处理方式
在`recoded_train.csv`基础上对数据做降维/embedding/过采样/..., 然后保存为新的结果文件放在`Dataset/processed/train_processed`下，然后在`data_fit.py`中修改：

```python
self.mode2dataset = {
   'normal': {'train': 'improved_logistic_imputed/improved_logistic_imputed_train_final.csv','test': 'improved_logistic_imputed/improved_logistic_imputed_test_final.csv'},
   '2class': {'train': 'improved_logistic_imputed/improved_logistic_imputed_train_final_2class.csv','test': 'improved_logistic_imputed/improved_logistic_imputed_test_final_2class.csv'},
   'my_mode': {'train':'path/to/train_dataset','test':'path/to/test_dataset'}
}
```

路径是相对于`train_processed/`的相对路径

## 项目结构

```
coursework/
├── config/
│   └── feature.json          # 特征配置文件
├── Dataset/
│   └── processed/
│       ├── train_processed/
│       │   ├── recoded_train.csv                    # 重编码训练数据
│       │   ├── improved_logistic_imputed/          # 改进逻辑回归填值结果
│       │   │   ├── improved_logistic_imputed_train_final.csv
│       │   │   ├── improved_logistic_imputed_test_final.csv
│       │   │   ├── imputation_report.json
│       │   │   └── model_info.json
│       │   └── modeling_results/                   # 模型训练结果
│       └── test_processed/
│           └── recoded_test.csv                    # 重编码测试数据
├── src/
│   ├── data_process.py                             # 数据预处理
│   ├── data_process_test.py                        # 测试集预处理
│   ├── improved_logistic_imputation.py             # 改进的逻辑回归填值
│   ├── data_fit.py                                 # 模型训练
│   └── data_visualization.py                       # 数据可视化
├── myutils/                                        # 自定义工具包
├── environment.yml                                 # 环境配置
├── setup.sh                                        # 快速安装脚本
├── Makefile                                        # 构建脚本
├── docs.md                                         # 详细文档
└── README.md                                       # 本文档
```

> Dataset/FeatureTabel_Ch.xlsx: 带中文解释 && gpt意见的FeatureTabel

## 特征配置文件说明 (config/feature.json)

feature.json包含了数据集的所有特征配置信息，主要包括以下字段：

### 全局配置
- `dataset_name`: 数据集名称
- `target_feature`: 目标特征（标签）名称
- `nan_values`: 所有可能的缺失值表示方式列表

### 特征配置 (features字段)
每个特征都包含以下信息：

1. 基本信息
   - `feature_id`: 特征ID（用于排序和引用）
   - `category`: 特征类别（demographic/categorical/identifier等）
   - `type`: 数据类型（categorical/numerical）
   - `description`: 特征描述（中文）
   - `iskeep`: 是否保留该特征
   - `process`: 处理方式（"normal"/"no"）

2. 编码信息（type==categorical特征）
   - `label_encoding`: 标签编码信息
     - `unique_values`: 所有唯一值列表
     - `encoding_mapping`: 值到编码的映射字典

3. 缺失值信息(type==categorical特征才有缺失)
   - `missing`: 是否存在缺失值
   - `missing_values_num`: 缺失值数量
   - `missing_values_p`: 缺失值比例
   - `missing_values`: 特征特有的缺失值列表
   - `missing_replace`: 缺失值替换值
   - `missing_in_test_num`: 测试集中的缺失数量

4. 数值范围（normalized前）
   - `max_value`: 特征最大值
   - `min_value`: 特征最小值

## 使用方法

### 原始数据可视化
```bash
conda activate 309
python src/data_visualization.py
```

### 测试改进的填值方法
```bash
python src/test_improved_imputation.py
```

## 常见问题

- **ImportError: No module named 'myutils'**
  - 进入myutils目录，运行`pip install -e .`
- **myutils目录为空**
  - 运行`git submodule update --init --recursive`
- **填值过程中出现迭代限制警告**
  - 这是正常现象，改进的方法会自动处理并继续填值
- **其他问题**请参考docs.md或运行`python test_setup.py`进行环境自检

## 更新日志

### v2.0 (2025-07-14)
- ✅ 实现了改进的逻辑回归缺失值填充方法
- ✅ 解决了"STOP: TOTAL NO. of ITERATIONS REACHED LIMIT"问题
- ✅ 支持训练集和测试集的统一填值
- ✅ 集成了完整的pipeline自动化流程
- ✅ 增强了错误处理和鲁棒性 
 