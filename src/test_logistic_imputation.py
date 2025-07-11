#!/usr/bin/env python3
"""
测试逻辑回归缺失值填补方法
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
import os

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__)))

from data_missing import MissingDataHandler

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)

def test_logistic_regression_imputation():
    """
    测试逻辑回归缺失值填补方法
    """
    logging.info("开始测试逻辑回归缺失值填补方法")
    
    # 初始化处理器
    handler = MissingDataHandler()
    
    # 加载数据
    handler.load_data(mode='recoded')
    
    # 分析缺失值
    handler.analyze_missing_values()
    
    # 获取训练数据
    train_data = handler.train_data.copy()
    
    # 显示缺失值情况
    missing_summary = train_data.isnull().sum()
    missing_features = missing_summary[missing_summary > 0]
    logging.info(f"有缺失值的特征数量: {len(missing_features)}")
    logging.info(f"缺失值最多的前5个特征:")
    for feature, count in missing_features.head().items():
        percentage = (count / len(train_data)) * 100
        logging.info(f"  {feature}: {count} ({percentage:.2f}%)")
    
    # 使用逻辑回归填补缺失值
    logging.info("开始逻辑回归填补...")
    imputed_data, model_info = handler.logistic_regression_impute(train_data)
    
    # 显示填补结果
    logging.info(f"填补完成，填补了 {len(model_info['features_imputed'])} 个特征")
    logging.info(f"填补的特征: {model_info['features_imputed']}")
    
    # 分析填补结果
    logging.info("开始分析填补结果...")
    analysis_result = handler.analyze_logistic_imputation(train_data)
    
    # 显示分析结果摘要
    logging.info(f"分析了 {len(analysis_result)} 个特征")
    logging.info(f"总共填补了 {analysis_result['missing_count'].sum()} 个缺失值")
    
    # 显示填补效果最好的特征（统计变化最小的）
    if len(analysis_result) > 0:
        # 计算统计变化
        analysis_result['mean_change'] = abs(analysis_result['mean_value(after)'] - analysis_result['mean_value(before)'])
        analysis_result['std_change'] = abs(analysis_result['std_value(after)'] - analysis_result['std_value(before)'])
        
        # 按均值变化排序
        best_features = analysis_result.nsmallest(5, 'mean_change')
        logging.info("填补效果最好的特征（均值变化最小）:")
        for _, row in best_features.iterrows():
            logging.info(f"  {row['feature']}: 均值变化 {row['mean_change']:.4f}, 标准差变化 {row['std_change']:.4f}")
    
    return imputed_data, model_info, analysis_result

def test_logistic_regression_impute_test():
    """
    测试对测试集进行逻辑回归填补
    """
    logging.info("开始测试测试集逻辑回归填补")
    
    # 初始化处理器
    handler = MissingDataHandler()
    
    # 加载测试数据（假设存在）
    test_data_path = 'Dataset/diabetic_data_test.csv'
    if not os.path.exists(test_data_path):
        logging.warning(f"测试数据文件不存在: {test_data_path}")
        logging.info("创建模拟测试数据进行演示...")
        
        # 创建模拟测试数据
        train_data = handler.load_data(mode='recoded')
        # 随机选择一些行作为测试数据
        test_data = handler.train_data.sample(n=100, random_state=42).copy()
        
        # 随机添加一些缺失值
        for col in test_data.columns:
            if test_data[col].dtype in ['object', 'category']:
                # 分类变量，随机设置10%为缺失
                mask = np.random.random(len(test_data)) < 0.1
                test_data.loc[mask, col] = np.nan
            else:
                # 数值变量，随机设置5%为缺失
                mask = np.random.random(len(test_data)) < 0.05
                test_data.loc[mask, col] = np.nan
        
        # 保存模拟测试数据
        test_data.to_csv(test_data_path, index=False)
        logging.info(f"创建模拟测试数据: {test_data_path}")
    else:
        test_data = pd.read_csv(test_data_path)
    
    # 显示测试数据缺失值情况
    missing_summary = test_data.isnull().sum()
    missing_features = missing_summary[missing_summary > 0]
    logging.info(f"测试集有缺失值的特征数量: {len(missing_features)}")
    
    # 使用训练好的模型对测试集进行填补
    try:
        imputed_test_data = handler.logistic_regression_impute_test(test_data)
        logging.info("测试集填补完成")
        
        # 显示填补结果
        missing_after = imputed_test_data.isnull().sum()
        missing_after_features = missing_after[missing_after > 0]
        logging.info(f"填补后测试集仍有缺失值的特征数量: {len(missing_after_features)}")
        
        if len(missing_after_features) > 0:
            logging.warning("以下特征在填补后仍有缺失值:")
            for feature, count in missing_after_features.items():
                logging.warning(f"  {feature}: {count}")
        
        return imputed_test_data
        
    except Exception as e:
        logging.error(f"测试集填补失败: {e}")
        return None

def compare_imputation_methods():
    """
    比较KNN和逻辑回归填补方法
    """
    logging.info("开始比较KNN和逻辑回归填补方法")
    
    # 初始化处理器
    handler = MissingDataHandler()
    
    # 加载数据
    handler.load_data(mode='recoded')
    train_data = handler.train_data.copy()
    
    # KNN填补
    logging.info("执行KNN填补...")
    knn_imputed = handler.knn_impute(train_data)
    knn_analysis = handler.analyse_knn_impute()
    
    # 逻辑回归填补
    logging.info("执行逻辑回归填补...")
    lr_imputed, lr_model_info = handler.logistic_regression_impute(train_data)
    lr_analysis = handler.analyze_logistic_imputation(train_data)
    
    # 比较结果
    logging.info("=== 填补方法比较 ===")
    logging.info(f"KNN填补特征数: {len(knn_analysis)}")
    logging.info(f"逻辑回归填补特征数: {len(lr_analysis)}")
    
    if len(knn_analysis) > 0 and len(lr_analysis) > 0:
        # 计算平均统计变化
        knn_mean_change = abs(knn_analysis['mean_value(after)'] - knn_analysis['mean_value(before)']).mean()
        lr_mean_change = abs(lr_analysis['mean_value(after)'] - lr_analysis['mean_value(before)']).mean()
        
        knn_std_change = abs(knn_analysis['std_value(after)'] - knn_analysis['std_value(before)']).mean()
        lr_std_change = abs(lr_analysis['std_value(after)'] - lr_analysis['std_value(before)']).mean()
        
        logging.info(f"KNN平均均值变化: {knn_mean_change:.4f}")
        logging.info(f"逻辑回归平均均值变化: {lr_mean_change:.4f}")
        logging.info(f"KNN平均标准差变化: {knn_std_change:.4f}")
        logging.info(f"逻辑回归平均标准差变化: {lr_std_change:.4f}")
        
        if knn_mean_change < lr_mean_change:
            logging.info("KNN在保持均值方面表现更好")
        else:
            logging.info("逻辑回归在保持均值方面表现更好")
            
        if knn_std_change < lr_std_change:
            logging.info("KNN在保持标准差方面表现更好")
        else:
            logging.info("逻辑回归在保持标准差方面表现更好")

def main():
    """
    主函数
    """
    logging.info("=== 逻辑回归缺失值填补测试 ===")
    
    try:
        # 测试训练集填补
        imputed_data, model_info, analysis_result = test_logistic_regression_imputation()
        
        # 测试测试集填补
        imputed_test_data = test_logistic_regression_impute_test()
        
        # 比较填补方法
        compare_imputation_methods()
        
        logging.info("所有测试完成")
        
    except Exception as e:
        logging.error(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 