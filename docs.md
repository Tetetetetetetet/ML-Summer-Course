# 糖尿病数据集分析项目开发文档

## 项目概述

本项目旨在对糖尿病数据集进行全面的特征分析和可视化，为后续的机器学习模型训练提供数据预处理支持。

### 项目结构
```
coursework/
├── Dataset/                    # 原始数据集
│   ├── diabetic_data_training.csv
│   ├── diabetic_data_test.csv
│   ├── FeatureTable.xlsx
│   └── IDS_mapping.csv
├── src/                       # 源代码目录
│   ├── data_visualization.py  # 数据可视化主程序
│   ├── feature.json          # 特征配置文件
│   ├── visualizations/       # 可视化输出目录
│   └── modify_feature_json.py # 特征文件修改工具
├── myutils/                   # 工具包
│   └── myutils/
│       ├── __init__.py
│       ├── genie.py
│       └── model.py
└── docs.md                   # 本文档
```

## 特征配置文件 (feature.json)

### 文件结构
```json
{
  "dataset_name": "糖尿病数据集",
  "target_feature": "readmitted",
  "features": {
    "feature_name": {
      "feature_id": 0,
      "category": "demographic|categorical|numeric|diagnosis|medication|identifier",
      "type": "categorical|continuous",
      "description": "特征描述",
      "visualization": "可视化类型",
      "label_encoding": {
        "unique_values": ["值1", "值2", ...],
        "encoding_mapping": {
          "值1": 0,
          "值2": 1,
          ...
        }
      }
    }
  }
}
```

### 特征分类
- **demographic**: 人口统计学特征 (race, gender, age, weight)
- **categorical**: 分类特征 (admission_type_id, discharge_disposition_id, etc.)
- **numeric**: 数值型特征 (time_in_hospital, num_lab_procedures, etc.)
- **diagnosis**: 诊断特征 (diag_1, diag_2, diag_3)
- **medication**: 药物特征 (metformin, insulin, etc.)
- **identifier**: 标识符特征 (encounter_id, patient_nbr)

## 核心模块

### DataVisualizer 类

#### 类初始化
```python
class DataVisualizer:
    def __init__(self):
        self.dataset_path = Path(__file__).parent.parent / "Dataset"
        self.train_data = None
        self.test_data = None
        self.feature_file = Path(__file__).parent.parent / "feature.json"
        self.feature_json = read_jsonl(str(self.feature_file))
```

#### 主要方法

##### 1. 数据加载
```python
def load_data(self):
    """加载训练集和测试集数据"""
```

##### 2. 特征处理
```python
def get_demographic_features(self):
    """从feature.json中获取demographic特征列表"""

def process_demographic_features(self):
    """处理demographic特征，包括体重缺失值处理"""
```

##### 3. 可视化方法
```python
def visualize_demographic_features_histogram(self):
    """为demographic特征创建直方图，标注数量和比例"""

def visualize_numeric_features(self):
    """可视化数值型特征"""

def visualize_categorical_features(self):
    """可视化类别型特征"""

def visualize_medication_features(self):
    """可视化药物特征"""

def visualize_diagnosis_features(self):
    """可视化诊断特征"""

def visualize_target_feature(self):
    """可视化目标特征"""

def visualize_correlations(self):
    """可视化特征相关性"""

def visualize_missing_values(self):
    """可视化缺失值情况"""
```

##### 4. 编码处理
```python
def create_label_encoding(self):
    """为demographic特征创建标签编码（0,1,2,...序列）"""

def update_feature_json(self, label_encoding_data):
    """更新feature.json文件，以name为key写入features"""
```

## 特征处理规范

### Demographic特征处理

#### 1. Race (种族)
- **原始值**: ['Caucasian', 'AfricanAmerican', '?', 'Other', 'Hispanic', 'Asian']
- **标签编码**: 0-5
- **缺失值处理**: '?' 作为有效类别处理

#### 2. Gender (性别)
- **原始值**: ['Female', 'Male', 'Unknown/Invalid']
- **标签编码**: 0-2
- **缺失值处理**: 'Unknown/Invalid' 作为有效类别处理

#### 3. Age (年龄组)
- **原始值**: ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)']
- **标签编码**: 0-9 (特殊处理：[0-10) 编码为 0)
- **排序规则**: 按年龄组顺序排列

#### 4. Weight (体重)
- **原始值**: ['Missing', 'Available']
- **标签编码**: 0-1
- **缺失值处理**: 将 '?' 统一处理为 'Missing'，其他值处理为 'Available'

### 编码规范

#### 标签编码 (Label Encoding)
- 将类别特征转换为 0, 1, 2, ... 的序列
- 保持类别间的相对顺序
- 特殊处理：age特征中 [0-10) 编码为 0

#### 缺失值处理
- 体重特征：'?' → 'Missing'
- 其他特征：保留原始缺失值标记

## 可视化规范

### 图表类型
- **直方图**: 用于显示特征分布
- **饼图**: 用于显示比例分布
- **条形图**: 用于显示类别计数
- **热力图**: 用于显示相关性
- **水平条形图**: 用于显示类别特征分布

### 标注要求
- 显示具体数量和百分比
- 添加总数信息
- 使用中文标签
- 合理的颜色搭配

### 输出规范
- 保存路径: `src/visualizations/`
- 文件格式: PNG
- 分辨率: 300 DPI
- 文件名: `{feature_category}_features.png`

## 开发规范

### 代码风格
- 使用中文注释
- 函数名使用下划线命名法
- 类名使用驼峰命名法
- 变量名使用下划线命名法

### 文件组织
- 所有源代码放在 `src/` 目录下
- 配置文件放在项目根目录
- 输出文件放在对应的输出目录

### 错误处理
- 使用 try-except 处理文件读取错误
- 检查数据完整性
- 提供有意义的错误信息

### 性能优化
- 避免重复读取大文件
- 使用适当的数据结构
- 及时释放内存

## API 文档

### DataVisualizer 类 API

#### 初始化
```python
visualizer = DataVisualizer()
```

#### 主要方法调用
```python
# 创建所有可视化
visualizer.create_all_visualizations()

# 处理demographic特征
visualizer.process_and_visualize_demographic()

# 单独创建可视化
visualizer.visualize_demographic_features_histogram()
visualizer.visualize_numeric_features()
visualizer.visualize_categorical_features()
```

### 配置文件 API

#### 读取特征配置
```python
from myutils import read_jsonl
feature_json = read_jsonl("feature.json")
```

#### 获取特定类别特征
```python
demographic_features = [name for name, info in feature_json['features'].items() 
                       if info['category'] == 'demographic']
```

## 开发计划

### 已完成
- [x] Demographic特征处理
- [x] 体重缺失值处理
- [x] 标签编码实现
- [x] 直方图可视化
- [x] feature.json结构优化

### 待完成
- [ ] Categorical特征处理
- [ ] Numeric特征处理
- [ ] Diagnosis特征处理
- [ ] Medication特征处理
- [ ] 特征相关性分析
- [ ] 缺失值分析
- [ ] 数据质量报告

### 下一步计划
1. 实现categorical特征的处理和可视化
2. 完善numeric特征的分析
3. 添加特征选择功能
4. 实现数据预处理流水线

## 注意事项

1. **路径处理**: 所有相对路径都以项目根目录为基准
2. **编码问题**: 确保中文字符正确显示
3. **内存管理**: 大数据集处理时注意内存使用
4. **版本控制**: 重要修改前备份配置文件
5. **测试**: 新功能添加后进行充分测试

## 故障排除

### 常见问题
1. **中文显示乱码**: 检查字体设置
2. **路径错误**: 确认工作目录
3. **内存不足**: 分批处理大数据集
4. **编码错误**: 检查JSON文件格式

### 调试技巧
- 使用print语句输出中间结果
- 检查数据类型和形状
- 验证配置文件格式
- 查看错误堆栈信息
