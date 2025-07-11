#!/usr/bin/env python3
"""
测试data_process_test.py的完整功能
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
import os

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__)))

from data_process_test import DataProcessTest

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)

def test_complete_data_process_test():
    """
    测试DataProcessTest的完整功能
    """
    logging.info("开始测试DataProcessTest的完整功能")
    
    # 初始化处理器
    handler = DataProcessTest()
    
    # 1. 加载数据
    handler.load_data()
    logging.info(f"原始测试数据形状: {handler.test_data.shape}")
    
    # 2. 配置特征
    handler.config_features()
    
    # 3. 清理无效数据
    handler.clean_invalid_data()
    
    # 4. 标记极端特征
    handler.mark_extream_features()
    
    # 5. 删除特征
    handler.drop_features()
    
    # 6. 转换缺失值
    handler.transfer_all_nan()
    
    # 7. 分析缺失值
    missing_stats = handler.analyze_missing_values()
    
    # 8. 编码测试数据
    handler.encode_test_data()
    
    # 9. 检查重新编码后的数据
    handler.check_recoded_data()
    
    # 10. 保存结果
    handler.save_train_data('recoded_test')
    handler.save_results(missing_stats)
    
    logging.info("DataProcessTest完整功能测试完成")
    return handler, missing_stats

def compare_train_test_processing():
    """
    比较训练集和测试集的处理结果
    """
    logging.info("比较训练集和测试集的处理结果")
    
    # 读取训练集处理结果
    train_data_path = 'Dataset/processed/train_processed/recoded_train.csv'
    test_data_path = 'Dataset/processed/test_processed/recoded_test.csv'
    
    if not os.path.exists(train_data_path):
        logging.error(f"训练集处理结果不存在: {train_data_path}")
        return
    
    if not os.path.exists(test_data_path):
        logging.error(f"测试集处理结果不存在: {test_data_path}")
        return
    
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
    
    logging.info(f"训练集形状: {train_data.shape}")
    logging.info(f"测试集形状: {test_data.shape}")
    
    # 检查列名是否一致
    train_columns = set(train_data.columns)
    test_columns = set(test_data.columns)
    
    common_columns = train_columns & test_columns
    train_only = train_columns - test_columns
    test_only = test_columns - train_columns
    
    logging.info(f"共同特征数: {len(common_columns)}")
    logging.info(f"训练集独有特征: {len(train_only)}")
    logging.info(f"测试集独有特征: {len(test_only)}")
    
    if train_only:
        logging.warning(f"训练集独有特征: {list(train_only)}")
    if test_only:
        logging.warning(f"测试集独有特征: {list(test_only)}")
    
    # 检查诊断特征的处理
    diagnosis_features = ['diag_1', 'diag_2', 'diag_3']
    for feature in diagnosis_features:
        if feature in common_columns:
            train_unique = set(train_data[feature].dropna().unique())
            test_unique = set(test_data[feature].dropna().unique())
            
            logging.info(f"\n=== {feature} 处理结果比较 ===")
            logging.info(f"训练集唯一值: {sorted(train_unique)}")
            logging.info(f"测试集唯一值: {sorted(test_unique)}")
            
            # 检查是否有测试集独有的值
            test_only_values = test_unique - train_unique
            if test_only_values:
                logging.warning(f"{feature} 测试集独有值: {test_only_values}")
            
            # 检查缺失值情况
            train_missing = train_data[feature].isna().sum()
            test_missing = test_data[feature].isna().sum()
            logging.info(f"训练集缺失值: {train_missing}")
            logging.info(f"测试集缺失值: {test_missing}")

def analyze_test_processing_results():
    """
    分析测试集处理结果
    """
    logging.info("分析测试集处理结果")
    
    # 读取处理报告
    report_path = 'Dataset/processed/test_processed/test_processing_report.json'
    if os.path.exists(report_path):
        import json
        with open(report_path, 'r') as f:
            report = json.load(f)
        
        logging.info("测试集处理报告:")
        logging.info(f"  处理的特征数: {report['test_data_info']['features_processed']}")
        logging.info(f"  有缺失值的特征数: {report['missing_value_summary']['total_features_with_missing']}")
        logging.info(f"  总缺失值数: {report['missing_value_summary']['total_missing_values']}")
        logging.info(f"  平均缺失率: {report['missing_value_summary']['average_missing_percentage']:.2f}%")
        logging.info(f"  编码的分类特征数: {report['encoding_summary']['categorical_features_encoded']}")
    
    # 读取缺失值统计
    missing_stats_path = 'Dataset/processed/test_processed/test_missing_stats.csv'
    if os.path.exists(missing_stats_path):
        missing_stats = pd.read_csv(missing_stats_path, index_col=0)
        
        logging.info("\n缺失值最多的前10个特征:")
        top_missing = missing_stats.nlargest(10, 'total_missing')
        for feature, row in top_missing.iterrows():
            logging.info(f"  {feature}: {row['total_missing']} ({row['missing_percentage']:.2f}%)")

def main():
    """
    主函数
    """
    logging.info("=== DataProcessTest完整功能测试 ===")
    
    try:
        # 测试完整功能
        handler, missing_stats = test_complete_data_process_test()
        
        # 比较训练集和测试集处理结果
        compare_train_test_processing()
        
        # 分析测试集处理结果
        analyze_test_processing_results()
        
        logging.info("所有测试完成")
        
    except Exception as e:
        logging.error(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 