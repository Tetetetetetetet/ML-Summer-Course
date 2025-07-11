# 糖尿病数据集分析项目

本项目对糖尿病数据集进行特征分析和可视化，为机器学习模型训练提供数据预处理支持。

## 环境配置

### 快速安装（推荐）

```bash
./setup.sh
```

### 手动安装

```bash
# 1. 创建环境
conda env create -f environment.yml
conda activate diabetes_analysis

# 2. 安装自定义工具包
cd myutils
pip install -e .
```

## 项目结构

```
coursework/
├── config/
│   └── feature.json  # 特征配置文件
├── Dataset/
├── src/
├── myutils/
├── environment.yml
├── setup.sh
├── docs.md
└── README.md
```

> Dataset/FeatureTabel_Ch.xlsx: 带中文解释 && gpt意见的FeatureTabel

## 特征配置文件说明 (feature.json)

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

4. 数值范围（normalized前）
   - `max_value`: 特征最大值
   - `min_value`: 特征最小值

## 使用方法

### 原始数据可视化
```bash
conda activate diabetes_analysis
python src/data_visualization.py
```

## 数据处理流程

### 特征重编码(src/data_process.py)：
1. 对原始数据中的离散feature重新编码(类别编码,0,1,...)
2. 找出所有等价于缺失的值，转换为None -> Dataset/processed/missing_replaced_train.csv
3. 重新编码(类别编码,0,1,...) -> Dataset/processed/recoded_train
4. 归一化 -> Dataset/processed/normalizaed_train.csv
(5.) (todo) 对某些特征做one-hot编码？

### 缺失值处理(src/data_missing.py)：
1. 主要步骤：
   - 去除完全缺失的特征
   - 对每个"≤5%缺失"的特征中存在缺失值的样本进行PPCQ分析

2. PPCQ分析流程：
   a. 处理"50-50特征"（缺失值比例接近50%的特征）：
      - 分析"≤5%缺失"特征对这些特征的重要度
      - 对不重要的特征：进行急救方向探空
      - 对重要的特征：与其他特征进行关联分析
   
   b. 处理含有"50-50"特征缺失的样本：
      - 分析缺失难度的重要度
      - 对有医学显著性的条件使用逻辑回归
      
3. 缺失值填充决策：
   - 判断特征是否与其他"50-50"特征相关：
     - 相关：按照特征分布进行预估填充
     - 不相关：待定
   - 判断特征是否存在相关性：
     - 存在：使用回归模型预测缺失值
     - 不存在：待定

4. 输出文件：
   - complete_features/：每个特征的完整数据集
   - all_features_complete.csv：所有特征完整的数据集
   - missing_stats.json：缺失值统计信息

## 常见问题

- ImportError: No module named 'myutils'
  - 进入myutils目录，运行`pip install -e .`
- myutils目录为空
  - 运行`git submodule update --init --recursive`
- 其他问题请参考docs.md或运行`python test_setup.py`进行环境自检 
 