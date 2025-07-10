# 糖尿病数据集分析项目开发文档

## 项目概述

本项目对糖尿病数据集进行特征分析和可视化，为机器学习模型训练提供数据预处理支持。

### 项目结构
```
coursework/
├── config/                    # 配置文件目录
│   ├── feature.json          # 特征配置文件
│   └── id_mapping.json       # ID映射配置文件
├── Dataset/                   # 原始数据集
│   └── processed/            # 处理后的数据
│       ├── recoded_train.csv # 重编码后的训练数据
│       ├── normalized_train.csv # 归一化后的训练数据
│       └── visualization/    # 特征可视化图表
├── src/                      # 源代码目录
│   ├── data_process.py      # 数据处理主程序
│   └── data_visualization.py # 数据可视化主程序
└── docs.md                   # 本文档
```

## 数据处理流程

### 1. 特征重编码
- 根据feature.json中的配置进行特征重编码
- 对分类特征进行标签编码
- 处理缺失值
- 输出：recoded_train.csv

### 2. 数据归一化
- 对所有保留的特征进行归一化处理（包括分类特征和数值特征）
- 将特征值归一化到[0,1]区间
- 记录每个特征的最大值和最小值到feature.json
- 对于所有值相同的特征，设置为0
- 输出：normalized_train.csv

## 特征配置文件 (feature.json)

### 文件结构
```json
{
    "feature_name": {
        "feature_id": 0,
        "category": "demographic|categorical|numeric|diagnosis|medication|identifier",
        "type": "categorical|continuous",
        "description": "特征描述",
        "visualization": "可视化类型",
        "label_encoding": {
            "unique_values": ["值1", "值2", ...],
            "encoding_mapping": {"值1": 0, "值2": 1, ...}
        },
        "iskeep": true|false,
        "max_value": 1.0,  // 归一化后添加
        "min_value": 0.0   // 归一化后添加
    }
}
```

## 特征处理规范

### Demographic特征
- **race**: 种族 (6个值) - 完整显示
- **gender**: 性别 (3个值) - 完整显示  
- **age**: 年龄组 (10个值) - 完整显示
- **weight**: 体重 (2个值) - 完整显示

### Categorical特征
- **admission_type_id**: 入院类型 (8个值) - 完整显示
- **discharge_disposition_id**: 出院处置 (26个值) - 完整显示
- **admission_source_id**: 入院来源 (17个值) - 完整显示
- **payer_code**: 支付方代码 (18个值) - 完整显示
- **medical_specialty**: 医疗专业 (73个值) - **⚠️ 只显示前20个**
- **max_glu_serum**: 最大血糖血清 (3个值) - 完整显示
- **A1Cresult**: A1C结果 (3个值) - 完整显示
- **change**: 药物变化 (2个值) - 完整显示
- **diabetesMed**: 糖尿病药物 (2个值) - 完整显示

### 诊断特征
- **diag_1**: 主要诊断 (711个值) - **⚠️ 只显示前30个**
- **diag_2**: 次要诊断 (731个值) - **⚠️ 只显示前30个**  
- **diag_3**: 第三诊断 (776个值) - **⚠️ 只显示前30个**

### 数值型特征
- **time_in_hospital**: 住院时间
- **num_lab_procedures**: 实验室检查次数
- **num_procedures**: 手术次数
- **num_medications**: 药物数量
- **number_outpatient**: 门诊次数
- **number_emergency**: 急诊次数
- **number_inpatient**: 住院次数
- **number_diagnoses**: 诊断数量

## 可视化规范

### 显示限制
- **高基数特征** (>20个取值): 只显示前20个最常见的值
- **诊断特征** (>30个取值): 只显示前30个最常见的值
- **其他特征**: 完整显示所有取值

### 输出规范
- 保存路径: `src/visualizations/`
- 文件格式: PNG, 300 DPI
- 不显示图形: 只保存文件，不调用 `plt.show()`
- 标注要求: 显示数量和百分比，添加总数信息

## 核心模块

### DataProcess 类
```python
class DataProcess:
    def __init__(self):
        # 初始化数据路径和配置文件
    
    def recode_train_data(self):
        # 重编码训练数据
    
    def normalize_data(self):
        # 对所有特征进行归一化处理
        # 更新feature.json中的最大最小值
    
    def save_train_data(self, name):
        # 保存处理后的数据
```

### DataVisualizer 类
```python
class DataVisualizer:
    def __init__(self):
        # 初始化数据路径和配置文件
    
    def create_all_visualizations(self):
        # 创建所有可视化
    
    def process_and_visualize_demographic(self):
        # 处理demographic特征
    
    def process_categorical_features(self):
        # 处理categorical特征
```

## 生成文件列表

### 可视化输出文件
- **Demographic**: `demographic_features_histogram.png`
- **Categorical**: 11个单独的特征分布图
- **诊断特征**: 3个单独的诊断分布图
- **其他**: 数值型、药物、目标特征、相关性、缺失值等

### 配置文件
- `config/feature.json` - 特征配置文件（包含标签编码和映射信息）
- `config/id_mapping.json` - ID映射配置文件

## 已完成功能
- [x] 特征重编码
- [x] 数据归一化
- [x] 特征可视化
- [x] 配置文件自动更新
- [x] 数据验证和检查

## 注意事项

1. **数据处理顺序**: 必须先进行重编码，再进行归一化
2. **特征选择**: 通过feature.json中的iskeep字段控制
3. **归一化处理**: 
   - 所有特征都进行归一化，包括分类特征
   - 特征的原始范围保存在feature.json中
   - 所有值相同的特征会被设置为0
4. **配置文件**: 
   - feature.json会自动更新，包含归一化信息
   - 确保有写入权限

## 故障排除

### 常见问题
1. **找不到重编码数据**: 确保先运行recode_train_data()
2. **归一化失败**: 检查特征类型和数值范围
3. **配置文件更新失败**: 检查文件权限和路径
4. **特征缺失**: 检查iskeep标记和特征名称匹配

### 使用示例
```python
dp = DataProcess()
dp.load_train_data()
dp.load_features_config()
dp.recode_train_data()
dp.save_train_data('recoded_train')
dp.normalize_data()  # 自动保存normalized_train.csv
```
