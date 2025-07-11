#!/usr/bin/env python3
"""
测试诊断特征缺失值处理（保持为None）
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
import os

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__)))

from data_process import DataProcess

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)

def test_diagnosis_missing_values():
    """
    测试诊断特征缺失值处理
    """
    logging.info("开始测试诊断特征缺失值处理")
    
    # 初始化处理器
    dp = DataProcess()
    
    # 加载原始数据
    dp.load_train_data(mode='origin')
    
    # 显示原始诊断特征的缺失值情况
    diagnosis_features = ['diag_1', 'diag_2', 'diag_3']
    for feature in diagnosis_features:
        if feature in dp.train_data.columns:
            missing_count = dp.train_data[feature].isna().sum()
            total_count = len(dp.train_data)
            missing_percentage = (missing_count / total_count) * 100
            logging.info(f"原始 {feature} 缺失值: {missing_count}/{total_count} ({missing_percentage:.2f}%)")
    
    # 配置特征
    dp.config_features()
    
    # 清理无效数据
    dp.clean_invalid_data()
    
    # 标记极端特征
    dp.mark_extream_features()
    
    # 删除特征
    dp.drop_features()
    
    # 重新编码（跳过诊断特征）
    dp.reencode()
    
    # 转换缺失值
    dp.transfer_all_nan()
    
    # 重新编码训练数据（包括诊断特征分类）
    logging.info("开始重新编码训练数据，包括诊断特征分类...")
    dp.recode_train_data()
    
    # 检查分类后的诊断特征缺失值情况
    logging.info("分类后的诊断特征缺失值情况:")
    for feature in diagnosis_features:
        if feature in dp.train_data.columns:
            missing_count = dp.train_data[feature].isna().sum()
            total_count = len(dp.train_data)
            missing_percentage = (missing_count / total_count) * 100
            logging.info(f"分类后 {feature} 缺失值: {missing_count}/{total_count} ({missing_percentage:.2f}%)")
            
            # 显示非缺失值的分布
            non_null_data = dp.train_data[feature].dropna()
            if len(non_null_data) > 0:
                value_counts = non_null_data.value_counts().sort_index()
                logging.info(f"  {feature} 非缺失值分布:")
                for category, count in value_counts.items():
                    percentage = (count / len(non_null_data)) * 100
                    logging.info(f"    类别 {category}: {count} ({percentage:.2f}%)")
    
    # 保存结果
    dp.save_train_data('recoded_train_with_none_missing')
    
    logging.info("诊断特征缺失值处理测试完成")
    
    return dp

def verify_feature_config():
    """
    验证feature配置是否正确更新
    """
    logging.info("验证feature配置")
    
    from myutils import read_jsonl
    feature_json = read_jsonl('config/feature.json')
    
    diagnosis_features = ['diag_1', 'diag_2', 'diag_3']
    for feature in diagnosis_features:
        if feature in feature_json['features']:
            config = feature_json['features'][feature]
            logging.info(f"\n=== {feature} 配置 ===")
            logging.info(f"process: {config.get('process', 'N/A')}")
            logging.info(f"missing: {config.get('missing', 'N/A')}")
            logging.info(f"value_num: {config.get('value_num', 'N/A')}")
            
            if 'label_encoding' in config:
                unique_values = config['label_encoding'].get('unique_values', [])
                encoding_mapping = config['label_encoding'].get('encoding_mapping', {})
                logging.info(f"unique_values: {unique_values}")
                logging.info(f"encoding_mapping keys: {list(encoding_mapping.keys())}")

def main():
    """
    主函数
    """
    logging.info("=== 诊断特征缺失值处理测试 ===")
    
    try:
        # 测试诊断特征缺失值处理
        dp = test_diagnosis_missing_values()
        
        # 验证feature配置
        verify_feature_config()
        
        logging.info("所有测试完成")
        
    except Exception as e:
        logging.error(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 