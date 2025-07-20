#!/usr/bin/env python3
"""
详细演示多分类AUC计算原理
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def demonstrate_ovr_auc():
    """
    演示One-vs-Rest AUC计算过程
    """
    print("========== One-vs-Rest AUC 计算演示 ==========")
    
    # 创建示例数据
    np.random.seed(42)
    n_samples = 1000
    
    # 生成三分类数据
    X = np.random.randn(n_samples, 5)
    y = np.random.randint(0, 3, n_samples)
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 训练模型
    model = LogisticRegression(random_state=42, multi_class='ovr')
    model.fit(X_train, y_train)
    
    # 获取预测概率
    y_pred_proba = model.predict_proba(X_test)
    
    print(f"预测概率矩阵形状: {y_pred_proba.shape}")
    print(f"目标变量分布: {np.bincount(y_test)}")
    
    # 方法1：使用sklearn的自动计算
    print("\n方法1: sklearn自动计算")
    auc_auto = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
    print(f"自动计算的AUC (macro): {auc_auto:.4f}")
    
    # 方法2：手动实现One-vs-Rest
    print("\n方法2: 手动实现One-vs-Rest")
    
    unique_classes = np.unique(y_test)
    auc_per_class = []
    
    for i, class_id in enumerate(unique_classes):
        # 创建二分类标签
        y_binary = (y_test == class_id).astype(int)
        
        # 获取该类别的概率
        proba_class = y_pred_proba[:, i]
        
        # 计算该类别的AUC
        auc_class = roc_auc_score(y_binary, proba_class)
        auc_per_class.append(auc_class)
        
        print(f"类别{class_id}:")
        print(f"  二分类标签分布: {np.bincount(y_binary)}")
        print(f"  概率范围: [{proba_class.min():.3f}, {proba_class.max():.3f}]")
        print(f"  AUC: {auc_class:.4f}")
    
    # 手动计算macro average
    auc_macro_manual = np.mean(auc_per_class)
    print(f"\n手动计算的Macro AUC: {auc_macro_manual:.4f}")
    print(f"验证: {abs(auc_auto - auc_macro_manual) < 1e-6}")
    
    return auc_per_class, auc_auto

def demonstrate_different_averages():
    """
    演示不同的平均方式
    """
    print("\n========== 不同平均方式对比 ==========")
    
    # 使用您的数据分布
    class_counts = {0: 47871, 1: 10245, 2: 31989}
    total_samples = sum(class_counts.values())
    
    # 模拟各类别的AUC分数
    auc_per_class = {0: 0.85, 1: 0.65, 2: 0.75}
    
    print("各类别AUC分数:")
    for class_id, auc in auc_per_class.items():
        count = class_counts[class_id]
        percentage = count / total_samples * 100
        print(f"  类别{class_id}: AUC={auc:.3f}, 样本数={count} ({percentage:.1f}%)")
    
    # 计算不同的平均方式
    # 1. Macro Average
    auc_macro = np.mean(list(auc_per_class.values()))
    
    # 2. Weighted Average
    auc_weighted = sum(count * auc_per_class[class_id] for class_id, count in class_counts.items()) / total_samples
    
    # 3. Micro Average (对于AUC，micro通常等于weighted)
    auc_micro = auc_weighted
    
    print(f"\n不同平均方式的结果:")
    print(f"Macro Average: {auc_macro:.4f}")
    print(f"Weighted Average: {auc_weighted:.4f}")
    print(f"Micro Average: {auc_micro:.4f}")
    
    print(f"\n差异分析:")
    print(f"Macro vs Weighted: {auc_weighted - auc_macro:.4f}")
    
    return auc_macro, auc_weighted

def explain_ovr_process():
    """
    详细解释OVR过程
    """
    print("\n========== OVR过程详细解释 ==========")
    
    # 示例数据
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    y_pred_proba = np.array([
        [0.8, 0.1, 0.1],  # 预测类别0的概率最高
        [0.2, 0.7, 0.1],  # 预测类别1的概率最高
        [0.1, 0.2, 0.7],  # 预测类别2的概率最高
        [0.9, 0.05, 0.05],
        [0.1, 0.8, 0.1],
        [0.05, 0.1, 0.85],
        [0.7, 0.2, 0.1],
        [0.1, 0.6, 0.3],
        [0.2, 0.1, 0.7]
    ])
    
    print("原始数据:")
    print(f"真实标签: {y_true}")
    print(f"预测概率矩阵:\n{y_pred_proba}")
    
    print("\nOne-vs-Rest转换过程:")
    
    for i, class_id in enumerate([0, 1, 2]):
        # 创建二分类标签
        y_binary = (y_true == class_id).astype(int)
        
        # 获取该类别的概率
        proba_class = y_pred_proba[:, i]
        
        print(f"\n类别{class_id} vs 其他:")
        print(f"  二分类标签: {y_binary}")
        print(f"  类别{class_id}概率: {proba_class}")
        
        # 计算AUC
        auc = roc_auc_score(y_binary, proba_class)
        print(f"  该类别的AUC: {auc:.4f}")
    
    # 计算macro average
    auc_values = []
    for i in range(3):
        y_binary = (y_true == i).astype(int)
        proba_class = y_pred_proba[:, i]
        auc = roc_auc_score(y_binary, proba_class)
        auc_values.append(auc)
    
    auc_macro = np.mean(auc_values)
    print(f"\nMacro Average AUC: {auc_macro:.4f}")
    
    # 验证与sklearn结果一致
    auc_sklearn = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
    print(f"sklearn结果: {auc_sklearn:.4f}")
    print(f"验证: {abs(auc_macro - auc_sklearn) < 1e-6}")

if __name__ == "__main__":
    # 运行所有演示
    auc_per_class, auc_auto = demonstrate_ovr_auc()
    auc_macro, auc_weighted = demonstrate_different_averages()
    explain_ovr_process()
    
    print("\n========== 总结 ==========")
    print("1. OVR方法为每个类别创建一个二分类器")
    print("2. 每个二分类器计算该类别的AUC")
    print("3. 最后通过平均方式合并所有类别的AUC")
    print("4. sklearn自动完成这个过程，我们只需要指定average参数") 