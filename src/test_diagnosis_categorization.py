#!/usr/bin/env python3
"""
测试诊断特征的分类处理
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

def test_diagnosis_categorization():
    """
    测试诊断特征的分类处理
    """
    logging.info("开始测试诊断特征分类处理")
    
    # 初始化处理器
    dp = DataProcess()
    
    # 加载原始数据
    dp.load_train_data(mode='origin')
    
    # 显示原始诊断特征的情况
    diagnosis_features = ['diag_1', 'diag_2', 'diag_3']
    for feature in diagnosis_features:
        if feature in dp.train_data.columns:
            unique_values = dp.train_data[feature].unique()
            logging.info(f"原始 {feature} 有 {len(unique_values)} 个唯一值")
            logging.info(f"前10个值: {unique_values[:10]}")
            
            # 显示一些示例的ICD-9编码
            sample_values = dp.train_data[feature].dropna().head(10)
            logging.info(f"{feature} 示例值: {sample_values.tolist()}")
    
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
    
    # 检查重新编码后的数据
    dp.check_recoded_data()
    
    # 显示分类后的诊断特征情况
    logging.info("分类后的诊断特征情况:")
    for feature in diagnosis_features:
        if feature in dp.train_data.columns:
            value_counts = dp.train_data[feature].value_counts().sort_index()
            logging.info(f"{feature} 分类结果:")
            for category, count in value_counts.items():
                percentage = (count / len(dp.train_data)) * 100
                logging.info(f"  类别 {category}: {count} ({percentage:.2f}%)")
    
    # 保存结果
    dp.save_train_data('recoded_train_with_diagnosis_categories')
    
    logging.info("诊断特征分类处理测试完成")
    
    return dp

def analyze_diagnosis_categories():
    """
    分析诊断特征分类的分布情况
    """
    logging.info("分析诊断特征分类分布")
    
    # 读取重新编码后的数据
    recoded_data_path = 'Dataset/processed/train_processed/recoded_train.csv'
    if not os.path.exists(recoded_data_path):
        logging.error(f"重新编码后的数据文件不存在: {recoded_data_path}")
        return
    
    data = pd.read_csv(recoded_data_path)
    
    diagnosis_features = ['diag_1', 'diag_2', 'diag_3']
    category_names = {
        0: '其他疾病',
        1: '循环系统疾病',
        2: '呼吸系统疾病', 
        3: '消化系统疾病',
        4: '糖尿病',
        5: '损伤和中毒',
        6: '肌肉骨骼系统疾病',
        7: '泌尿生殖系统疾病',
        8: '肿瘤'
    }
    
    for feature in diagnosis_features:
        if feature in data.columns:
            logging.info(f"\n=== {feature} 分类分布 ===")
            value_counts = data[feature].value_counts().sort_index()
            
            for category, count in value_counts.items():
                percentage = (count / len(data)) * 100
                category_name = category_names.get(category, f'未知类别{category}')
                logging.info(f"  {category} ({category_name}): {count} ({percentage:.2f}%)")
            
            # 检查是否有缺失值
            missing_count = data[feature].isna().sum()
            if missing_count > 0:
                logging.warning(f"  {feature} 仍有 {missing_count} 个缺失值")

def test_diagnosis_categorization_consistency():
    """
    测试诊断特征分类的一致性
    """
    logging.info("测试诊断特征分类的一致性")
    
    # 读取原始数据
    original_data = pd.read_csv('Dataset/diabetic_data_training.csv')
    
    # 读取重新编码后的数据
    recoded_data_path = 'Dataset/processed/train_processed/recoded_train.csv'
    if not os.path.exists(recoded_data_path):
        logging.error(f"重新编码后的数据文件不存在: {recoded_data_path}")
        return
    
    recoded_data = pd.read_csv(recoded_data_path)
    
    # 检查数据行数是否一致
    if len(original_data) != len(recoded_data):
        logging.error(f"数据行数不一致: 原始数据 {len(original_data)}, 重新编码后 {len(recoded_data)}")
        return
    
    diagnosis_features = ['diag_1', 'diag_2', 'diag_3']
    
    for feature in diagnosis_features:
        if feature in original_data.columns and feature in recoded_data.columns:
            logging.info(f"\n=== 检查 {feature} 分类一致性 ===")
            
            # 检查一些具体的ICD-9编码分类是否正确
            test_cases = [
                ('250.01', 5),  # 糖尿病 -> 类别5
                ('401.9', 2),   # 高血压 -> 循环系统疾病 -> 类别2
                ('486', 3),     # 肺炎 -> 呼吸系统疾病 -> 类别3
                ('V58.67', 1),  # V编码 -> 其他疾病 -> 类别1
                ('E11.9', 1),   # E编码 -> 其他疾病 -> 类别1
            ]
            
            for icd_code, expected_category in test_cases:
                # 在原始数据中找到这个编码的行
                mask = original_data[feature] == icd_code
                if mask.any():
                    # 检查重新编码后的分类
                    recoded_category = recoded_data.loc[mask, feature].iloc[0]
                    if recoded_category == expected_category:
                        logging.info(f"  ✓ {icd_code} -> 类别 {recoded_category} (正确)")
                    else:
                        logging.warning(f"  ✗ {icd_code} -> 类别 {recoded_category} (期望 {expected_category})")
                else:
                    logging.info(f"  - {icd_code} 在数据中未找到")

def main():
    """
    主函数
    """
    logging.info("=== 诊断特征分类处理测试 ===")
    
    try:
        # 测试诊断特征分类
        dp = test_diagnosis_categorization()
        
        # 分析分类分布
        analyze_diagnosis_categories()
        
        # 测试分类一致性
        test_diagnosis_categorization_consistency()
        
        logging.info("所有测试完成")
        
    except Exception as e:
        logging.error(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 