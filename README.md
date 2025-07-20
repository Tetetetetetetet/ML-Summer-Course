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

## train.sh

`train.sh` 是一个便捷的模型训练脚本，提供了灵活的参数配置选项。

### 参数说明

#### 基本参数
- **model**: 指定要训练的模型
  - `"all"`: 训练所有可用模型（RandomForest, GradientBoosting, LogisticRegression）
  - `"LogisticRegression"`: 仅训练逻辑回归模型
- `"RandomForest"`: 仅训练随机森林模型
- `"GradientBoosting"`: 仅训练梯度提升模型
- `"ResNet"`: 仅训练ResNet-like神经网络模型（需要TensorFlow）

- **mode**: 指定数据集模式
  - `"normal"`: 使用完整数据集
  - `"selected"`: 使用特征选择后的数据集
  - `"2class"`: 使用二分类数据集

- **k**: 特征选择数量参数
  - `"all"`: 保留所有特征（在已筛选的数据集基础上）
  - `整数`: 保留指定数量的特征（如 `10`, `20`, `50`）

- **feature_selector**: 特征选择方法
  - `"chi2"`: 使用卡方检验进行特征选择（适用于非负特征）
  - `"f_classif"`: 使用F检验进行特征选择（适用于数值特征）

- **isoversample**: 是否使用过采样
  - `true`: 使用SMOTE过采样处理类别不平衡
  - `false`: 不使用过采样

#### 实验命名规则
脚本会根据参数自动生成实验名称：
- 使用过采样：`{mode}_{k}_{model}_{feature_selector}_oversample`
- 不使用过采样：`{mode}_{k}_{model}_{feature_selector}_notoversample`

#### 特征选择说明
- **数据集预处理**: 所有数据集在进入模型训练前都经过了初步的特征筛选
- **k="all"的含义**: 在已筛选的数据集基础上保留所有特征，而不是原始数据集的所有特征
- **特征选择方法**: 可以选择不同的统计检验方法来评估特征重要性

### 使用示例

#### 1. 快速开始（使用默认参数）
```bash
./train.sh
```

#### 2. 训练特定模型
```bash
# 修改train.sh中的model参数
model="LogisticRegression"
./train.sh
```

#### 3. 使用特征选择
```bash
# 修改train.sh中的k参数
k=20  # 保留前20个特征
./train.sh
```

#### 4. 选择特征选择方法
```bash
# 修改train.sh中的feature_selector参数
feature_selector="f_classif"  # 使用F检验
./train.sh
```

#### 5. 启用过采样
```bash
# 修改train.sh中的isoversample参数
isoversample=true
./train.sh
```

#### 6. 完整自定义配置
```bash
# 修改train.sh中的所有参数
model="RandomForest"
mode="normal"
k=50
feature_selector="f_classif"
isoversample=true
./train.sh
```

#### 7. 使用ResNet神经网络模型
```bash
# 修改train.sh中的model参数
model="ResNet"
k=20
feature_selector="f_classif"
isoversample=false
./train.sh
```

### 输出结果

训练完成后，结果将保存在：
```
output/{exp_name}/
├── feature_scores_{feature_selector}.csv    # 特征选择分数
├── feature_scores_{feature_selector}.png    # 特征选择分数可视化
├── feature_importance.csv                   # 模型特征重要性（传统模型）
├── feature_importance.png                   # 特征重要性可视化（传统模型）
├── resnet_confusion_matrix.png              # ResNet混淆矩阵（仅ResNet模型）
├── resnet_training_history.png              # ResNet训练历史（仅ResNet模型）
├── resnet_results.json                      # ResNet详细结果（仅ResNet模型）
├── predictions.csv                          # 预测结果
├── best_model.pkl                           # 最佳模型文件（传统模型）
├── best_model.h5                            # 最佳模型文件（ResNet模型）
└── modeling_report.json                     # 建模报告
```

### 参数组合建议

#### 探索性分析
```bash
model="all"
mode="normal"
k="all"
isoversample=false
```

#### 特征选择实验
```bash
model="LogisticRegression"
mode="selected"
k=20
feature_selector="f_classif"
isoversample=false
```

#### 处理类别不平衡
```bash
model="all"
mode="normal"
k="all"
isoversample=true
```

#### 快速实验
```bash
model="LogisticRegression"
mode="selected"
k=10
feature_selector="chi2"
isoversample=false
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

## 特征选择方法详解

### chi2（卡方检验）
- **适用场景**: 非负特征（如计数、频率等）
- **原理**: 基于特征与目标变量之间的卡方统计量
- **优势**: 对分类特征效果好，计算速度快
- **限制**: 要求特征值非负

### f_classif（F检验）
- **适用场景**: 数值特征
- **原理**: 基于方差分析，计算特征与目标变量之间的F统计量
- **优势**: 适用于连续数值特征，理论基础扎实
- **限制**: 假设特征服从正态分布

### 选择建议
- **医疗数据**: 推荐使用 `f_classif`，因为医疗特征多为数值型
- **文本数据**: 推荐使用 `chi2`，因为文本特征通常为非负
- **混合数据**: 可以尝试两种方法，比较结果

### 特征选择流程
1. **预处理阶段**: 所有数据集都经过初步特征筛选
2. **特征评分**: 使用选定的统计方法计算每个特征的重要性分数
3. **特征排序**: 根据分数对特征进行排序
4. **特征选择**: 根据k参数选择前k个特征（k="all"时保留所有特征）
5. **结果保存**: 特征分数和可视化结果保存到输出目录

## ResNet神经网络模型

### 模型特点
- **架构**: ResNet-like全连接网络，包含残差连接
- **层数**: 3个残差块 + 2个全连接层
- **激活函数**: ReLU
- **正则化**: Dropout + BatchNormalization
- **输出**: 3分类softmax输出
- **接口兼容**: 完全兼容sklearn接口，可直接在data_fit.py中使用

### 优势
- **残差连接**: 缓解梯度消失问题，支持更深的网络
- **自动特征工程**: 通过多层网络自动学习特征表示
- **正则化**: 多种正则化技术防止过拟合
- **类别权重**: 自动处理类别不平衡问题
- **智能缓存**: 相同参数的模型只训练一次，自动加载已有模型

### 使用要求
- **TensorFlow**: 需要安装TensorFlow 2.x
- **内存**: 相比传统模型需要更多内存
- **训练时间**: 训练时间较长，建议使用GPU加速

### 参数说明
- **n**: 使用前n个特征（None表示使用所有特征）
- **epochs**: 训练轮数（默认100，可调整）
- **batch_size**: 批次大小（默认64）
- **validation_split**: 验证集比例（默认0.2）
- **class_weight**: 类别权重（自动计算）
- **need_train**: 是否需要训练（True训练并保存，False尝试加载已有模型）

### 特征维度检查
- **自动检查**: 自动检查特征数量与网络架构的兼容性
- **特征不足**: 如果特征数量小于指定数量，会报错并停止
- **特征过多**: 如果特征数量大于第一层维度(256)，会自动截取前256个特征

### 模型保存和加载
- **保存位置**: `output/resnet_models/` 目录
- **参数哈希**: 基于模型参数生成唯一标识
- **自动加载**: 相同参数的模型会自动加载，避免重复训练
- **保存内容**: 模型文件(.h5)、参数文件(.json)、标准化器(.pkl)、训练历史(.json)

### 使用示例
```bash
# 在train.sh中使用ResNet
model="ResNet"
k=20
feature_selector="f_classif"
./train.sh

# 或直接运行
python src/data_fit.py --model ResNet --k 20 --feature_selector f_classif
```
- ✅ 增强了错误处理和鲁棒性 
 