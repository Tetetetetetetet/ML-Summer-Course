#!/bin/bash

# 糖尿病数据集分析项目 - 快速安装脚本

echo "=== 糖尿病数据集分析项目安装脚本 ==="

# 检查conda是否安装
if ! command -v conda &> /dev/null; then
    echo "错误: 未找到conda，请先安装Anaconda或Miniconda"
    exit 1
fi

# 创建环境
echo "1. 创建conda环境..."
conda env create -f environment.yml

if [ $? -eq 0 ]; then
    echo "✓ 环境创建成功"
else
    echo "✗ 环境创建失败"
    exit 1
fi

# 激活环境
echo "2. 激活环境..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate diabetes_analysis

if [ $? -eq 0 ]; then
    echo "✓ 环境激活成功"
else
    echo "✗ 环境激活失败"
    exit 1
fi

# 安装myutils
echo "3. 安装myutils包..."
cd myutils
pip install -e .

if [ $? -eq 0 ]; then
    echo "✓ myutils安装成功"
else
    echo "✗ myutils安装失败"
    exit 1
fi

cd ..

# 验证安装
echo "4. 验证安装..."
python -c "import pandas, numpy, matplotlib, seaborn, sklearn, myutils; print('✓ 所有依赖包导入成功')"

if [ $? -eq 0 ]; then
    echo ""
    echo "=== 安装完成 ==="
    echo "使用方法:"
    echo "1. 激活环境: conda activate diabetes_analysis"
    echo "2. 运行分析: python src/data_visualization.py"
    echo "3. 查看文档: cat README.md"
else
    echo "✗ 依赖包验证失败"
    exit 1
fi 