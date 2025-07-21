#!/usr/bin/env python3
"""
测试F1-macro指标的正确性
与sklearn的f1_score进行对比验证
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
import sys
import os

# 添加src目录到路径
sys.path.append('src')

from resnet_model import f1_macro_metric

def test_f1_macro_metric():
    """测试F1-macro指标的正确性"""
    print("=== 测试F1-macro指标 ===")
    
    # 测试用例1：简单的3分类问题
    print("\n1. 测试用例1：简单的3分类问题")
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 1, 0, 1, 2])  # 有一个错误预测
    
    # 转换为one-hot编码
    y_true_onehot = tf.one_hot(y_true, depth=3)
    y_pred_onehot = tf.one_hot(y_pred, depth=3)
    
    # 计算sklearn的F1-macro
    sklearn_f1 = f1_score(y_true, y_pred, average='macro')
    
    # 计算我们的F1-macro
    our_f1 = f1_macro_metric(y_true_onehot, y_pred_onehot)
    
    print(f"真实标签: {y_true}")
    print(f"预测标签: {y_pred}")
    print(f"sklearn F1-macro: {sklearn_f1:.4f}")
    print(f"我们的 F1-macro: {our_f1:.4f}")
    print(f"差异: {abs(sklearn_f1 - our_f1):.6f}")
    
    # 测试用例2：不平衡数据集
    print("\n2. 测试用例2：不平衡数据集")
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 2, 2, 2])  # 类别0有5个，类别1有2个，类别2有3个
    y_pred = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])  # 有一些错误预测
    
    y_true_onehot = tf.one_hot(y_true, depth=3)
    y_pred_onehot = tf.one_hot(y_pred, depth=3)
    
    sklearn_f1 = f1_score(y_true, y_pred, average='macro')
    our_f1 = f1_macro_metric(y_true_onehot, y_pred_onehot)
    
    print(f"真实标签: {y_true}")
    print(f"预测标签: {y_pred}")
    print(f"sklearn F1-macro: {sklearn_f1:.4f}")
    print(f"我们的 F1-macro: {our_f1:.4f}")
    print(f"差异: {abs(sklearn_f1 - our_f1):.6f}")
    
    # 测试用例3：完美预测
    print("\n3. 测试用例3：完美预测")
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 2])  # 完美预测
    
    y_true_onehot = tf.one_hot(y_true, depth=3)
    y_pred_onehot = tf.one_hot(y_pred, depth=3)
    
    sklearn_f1 = f1_score(y_true, y_pred, average='macro')
    our_f1 = f1_macro_metric(y_true_onehot, y_pred_onehot)
    
    print(f"真实标签: {y_true}")
    print(f"预测标签: {y_pred}")
    print(f"sklearn F1-macro: {sklearn_f1:.4f}")
    print(f"我们的 F1-macro: {our_f1:.4f}")
    print(f"差异: {abs(sklearn_f1 - our_f1):.6f}")
    
    # 测试用例4：随机预测
    print("\n4. 测试用例4：随机预测")
    np.random.seed(42)
    y_true = np.random.randint(0, 3, 100)
    y_pred = np.random.randint(0, 3, 100)
    
    y_true_onehot = tf.one_hot(y_true, depth=3)
    y_pred_onehot = tf.one_hot(y_pred, depth=3)
    
    sklearn_f1 = f1_score(y_true, y_pred, average='macro')
    our_f1 = f1_macro_metric(y_true_onehot, y_pred_onehot)
    
    print(f"样本数量: {len(y_true)}")
    print(f"sklearn F1-macro: {sklearn_f1:.4f}")
    print(f"我们的 F1-macro: {our_f1:.4f}")
    print(f"差异: {abs(sklearn_f1 - our_f1):.6f}")
    
    # 测试用例5：符号张量兼容性
    print("\n5. 测试符号张量兼容性")
    try:
        # 创建符号张量
        y_true_sym = tf.keras.Input(shape=(3,))
        y_pred_sym = tf.keras.Input(shape=(3,))
        
        # 计算F1-macro
        f1_sym = f1_macro_metric(y_true_sym, y_pred_sym)
        
        print("✓ 符号张量兼容性测试通过")
        print(f"输出张量形状: {f1_sym.shape}")
        print(f"输出张量类型: {type(f1_sym)}")
        
    except Exception as e:
        print(f"✗ 符号张量兼容性测试失败: {e}")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_f1_macro_metric() 