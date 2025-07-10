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
│   └── feature.json
├── Dataset/
├── src/
├── myutils/
├── environment.yml
├── setup.sh
├── docs.md
└── README.md
```

> Dataset/FeatureTabel_Ch.xlsx: 带中文解释 && gpt意见的FeatureTabel

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
