
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
conda activate diabetes_analysis

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
数据预处理
```
make
```
预期结果：
```
[INFO] ==========保存最终结果==========
[INFO] 训练集保存到: Dataset/processed/train_processed/logistic_imputed/logistic_imputed_train_final.csv
[INFO] 测试集保存到: Dataset/processed/train_processed/logistic_imputed/logistic_imputed_test_final.csv
[INFO] 处理报告保存到: Dataset/processed/train_processed/logistic_imputed/imputation_report.json
[INFO] ==========逻辑回归填充完成==========
[INFO] 训练集: (90105, 29) -> (90105, 29)
[INFO] 测试集: (10009, 29) -> (10009, 29)
[INFO] 训练集缺失值: 103197 -> 0
[INFO] 测试集缺失值: 11411 -> 0
[INFO] ==========逻辑回归缺失值填充流程完成==========
```
训练，测试
1. 不使用过采样
```
make train
```
预期结果
```
[INFO] ==========save_model==========
[INFO] 模型保存到: Dataset/processed/train_processed/modeling_results/best_model_20250712_144301.pkl
[INFO] ==========generate_report==========
[INFO] 建模报告保存到: Dataset/processed/train_processed/modeling_results/modeling_report.json
[INFO] ==========建模总结==========
[INFO] 数据源: logistic_imputed
[INFO] 最佳模型: GradientBoosting
[INFO] 最佳准确率: 0.5821
[INFO] ==========建模流程完成==========
```
2. 使用过采样
```
make train OVERSAMPLE=true
```


### for windows
处理数据
```
python src/data_process.py
python src/data_process_test.py
python src/logistic_imputation_pipeline.py
```
训练
```
python src/data_fit.py
```

### 结果文件
- Dataset/train_processed/recoded_train.csv: 重新编码后的train数据集，缺失值记为None
- Dataset/test_processed/recoded_test.csv: 重新编码后的test数据集，缺失值记为None
- Dataset/processed/train_processed/logistic_imputed/logistic_imputed_train_final.csv: 逻辑回归填值后的train数据集，可直接用于训练
- Dataset/processed/train_processed/logistic_imputed/logistic_imputed_test_final.csv: 逻辑回归填值后的test数据集，可直接用于测试

## 如何修改
### 尝试新的数据预处理方式
在`recoded_train.csv`基础上对数据做降纬/embedding/过采样/..., 然后保存为新的结果文件放在`Dataset/processed/train_processed`下，然后在`data_fit` `46-48`行中类似如下
```python
self.mode2dataset = {
   'normal': {'train': 'logistic_imputed/logistic_imputed_train_final.csv','test': 'logistic_imputed/logistic_imputed_test_final.csv'},
   '2class': {'train': 'logistic_imputed/logistic_imputed_train_final_2class.csv','test': 'logistic_imputed/logistic_imputed_test_final_2class.csv'}
}
```
增加一项
```python
'my_mode': {'train':'path/to/train_dataset','test':'path/to/test_dataset'}
```
路径是相对于`train_processed/`的相对路径


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
conda activate diabetes_analysis
python src/data_visualization.py
```


## 常见问题

- ImportError: No module named 'myutils'
  - 进入myutils目录，运行`pip install -e .`
- myutils目录为空
  - 运行`git submodule update --init --recursive`
- 其他问题请参考docs.md或运行`python test_setup.py`进行环境自检 
 