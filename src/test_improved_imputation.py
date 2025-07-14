#!/usr/bin/env python3
"""
测试改进的逻辑回归缺失值填补方法
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
import os

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__)))

from improved_logistic_imputation import ImprovedLogisticImputation

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)

def create_test_data():
    """
    创建测试数据，模拟真实场景中的问题
    """
    np.random.seed(42)
    n_samples = 1000
    
    # 创建一些有问题的特征
    data = {
        'feature1': np.random.normal(0, 1, n_samples),  # 正常数值特征
        'feature2': np.random.normal(1000, 100, n_samples),  # 大尺度数值特征
        'feature3': np.random.choice(['A', 'B', 'C'], n_samples),  # 分类特征
        'feature4': np.random.choice(range(50), n_samples),  # 多类别数值特征
        'target': np.random.choice([0, 1], n_samples)  # 目标变量
    }
    
    df = pd.DataFrame(data)
    
    # 添加缺失值
    df.loc[100:200, 'feature1'] = np.nan
    df.loc[300:400, 'feature2'] = np.nan
    df.loc[500:600, 'feature3'] = np.nan
    df.loc[700:800, 'feature4'] = np.nan
    
    return df

def test_improved_imputation():
    """
    测试改进的填补方法
    """
    logging.info("开始测试改进的逻辑回归填补方法")
    
    # 创建测试数据
    test_data = create_test_data()
    logging.info(f"测试数据形状: {test_data.shape}")
    logging.info(f"缺失值情况:")
    for col in test_data.columns:
        missing_count = test_data[col].isna().sum()
        if missing_count > 0:
            percentage = (missing_count / len(test_data)) * 100
            logging.info(f"  {col}: {missing_count} ({percentage:.2f}%)")
    
    # 初始化改进的处理器
    handler = ImprovedLogisticImputation()
    
    # 进行填补
    logging.info("开始填补...")
    imputed_data = handler.impute_dataset(test_data)
    
    # 验证结果
    final_missing = imputed_data.isnull().sum().sum()
    logging.info(f"填补后剩余缺失值: {final_missing}")
    
    if final_missing == 0:
        logging.info("✅ 填补成功！所有缺失值都被填补")
    else:
        logging.warning(f"⚠️ 仍有 {final_missing} 个缺失值")
    
    # 分析填补结果
    logging.info("填补结果分析:")
    for col in test_data.columns:
        if test_data[col].isna().sum() > 0:
            original_unique = test_data[col].dropna().nunique()
            imputed_unique = imputed_data[col].nunique()
            logging.info(f"  {col}: 原始唯一值 {original_unique} -> 填补后唯一值 {imputed_unique}")
    
    return imputed_data

def compare_with_original():
    """
    与原始方法比较
    """
    logging.info("==========与原始方法比较==========")
    
    # 加载真实数据
    data_path = Path('Dataset/processed/train_processed/recoded_train.csv')
    if not data_path.exists():
        logging.error(f"真实数据文件不存在: {data_path}")
        return
    
    # 只加载部分数据进行测试
    data = pd.read_csv(data_path, nrows=1000)
    logging.info(f"加载真实数据样本: {data.shape}")
    
    # 分析缺失值
    missing_summary = data.isnull().sum()
    missing_features = missing_summary[missing_summary > 0]
    logging.info(f"有缺失值的特征: {list(missing_features.index)}")
    
    # 使用改进的方法
    handler = ImprovedLogisticImputation()
    imputed_data = handler.impute_dataset(data)
    
    # 验证结果
    final_missing = imputed_data.isnull().sum().sum()
    logging.info(f"改进方法填补后剩余缺失值: {final_missing}")
    
    return imputed_data

def main():
    """主函数"""
    logging.info("==========改进的逻辑回归填补测试开始==========")
    
    try:
        # 测试1: 使用模拟数据
        logging.info("测试1: 使用模拟数据")
        test_improved_imputation()
        
        # 测试2: 使用真实数据
        logging.info("测试2: 使用真实数据")
        compare_with_original()
        
        logging.info("==========所有测试完成==========")
        
    except Exception as e:
        logging.error(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 