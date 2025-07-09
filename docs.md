# 糖尿病数据集分析项目开发文档

## 项目概述

本项目对糖尿病数据集进行特征分析和可视化，为机器学习模型训练提供数据预处理支持。

### 项目结构
```
coursework/
├── Dataset/                    # 原始数据集
├── src/                       # 源代码目录
│   ├── data_visualization.py  # 数据可视化主程序
│   ├── visualizations/       # 可视化输出目录
│   └── feature.json          # 特征配置文件
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
        "encoding_mapping": {"值1": 0, "值2": 1, ...}
      }
    }
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
- `feature.json` - 特征配置文件（包含标签编码和映射信息）

## 已完成功能
- [x] Demographic特征处理
- [x] Categorical特征处理
- [x] 诊断特征可视化
- [x] 数值型特征可视化
- [x] 药物特征可视化
- [x] 特征相关性分析
- [x] 缺失值分析
- [x] 移除plt.show()，只保存图片
- [x] 总数验证和百分比显示

## 注意事项

1. **高基数特征**: medical_specialty、diag_1/2/3 因取值过多，只显示部分数据
2. **路径处理**: 所有相对路径都以项目根目录为基准
3. **编码问题**: 确保中文字符正确显示
4. **内存管理**: 大数据集处理时注意内存使用

## 故障排除

### 常见问题
1. **中文显示乱码**: 检查字体设置
2. **路径错误**: 确认工作目录
3. **图片不显示**: 已移除plt.show()，只保存文件
4. **高基数特征显示不全**: 这是正常行为，只显示最常见的值
