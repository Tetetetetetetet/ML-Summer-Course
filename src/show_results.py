from myutils import read_jsonl
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

def show_results(results_path='output/network_data-k_all-model_ResNet-bs_128-ep_1000-fs_None-oversample/predictions.csv',
                true_labels_path='Dataset/processed/train_processed/network_test.csv'):
    """
    分析预测结果，计算各个类别的TP、TN、FP、FN
    
    Args:
        results_path: 预测结果文件路径
        true_labels_path: 真实标签文件路径
    """
    # 读取预测结果
    results = pd.read_csv(results_path)
    print(f"预测结果文件列名: {list(results.columns)}")
    
    # 读取真实标签
    true_data = pd.read_csv(true_labels_path)
    print(f"真实标签文件列名: {list(true_data.columns)}")
    
    # 检查预测结果中的列
    if 'predicted_readmitted' not in results.columns:
        print(f"错误：预测结果文件中缺少 'predicted_readmitted' 列")
        print(f"可用的列: {list(results.columns)}")
        return
    
    # 获取预测标签和真实标签
    y_pred = results['predicted_readmitted']
    y_true = true_data['readmitted']
    
    # 确保两个数组长度一致
    min_len = min(len(y_pred), len(y_true))
    y_pred = y_pred[:min_len]
    y_true = y_true[:min_len]
    
    print(f"样本数量: {min_len}")
    
    # 获取所有类别
    all_classes = sorted(list(set(y_true) | set(y_pred)))
    print(f"类别: {all_classes}")
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=all_classes)
    
    # 创建结果DataFrame
    results_df = []
    
    for i, class_name in enumerate(all_classes):
        # 对于每个类别，计算二分类的混淆矩阵
        tp = cm[i, i]  # 真正例
        fp = cm[:, i].sum() - tp  # 假正例
        fn = cm[i, :].sum() - tp  # 假负例
        tn = cm.sum() - tp - fp - fn  # 真负例
        
        # 计算指标
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        
        results_df.append({
            '类别': class_name,
            'TP': tp,
            'TN': tn,
            'FP': fp,
            'FN': fn,
            '精确率': f"{precision:.4f}",
            '召回率': f"{recall:.4f}",
            'F1分数': f"{f1:.4f}"
        })
    
    # 创建DataFrame
    results_df = pd.DataFrame(results_df)
    
    print("\n" + "=" * 80)
    print("各类别的TP、TN、FP、FN统计")
    print("=" * 80)
    print(results_df.to_string(index=False))
    
    # 计算总体指标
    total_tp = sum(cm[i, i] for i in range(len(all_classes)))
    total_samples = cm.sum()
    accuracy = total_tp / total_samples if total_samples > 0 else 0
    
    print(f"\n总体准确率: {accuracy:.4f}")
    print(f"总样本数: {total_samples}")
    
    # 显示混淆矩阵
    print(f"\n混淆矩阵:")
    print("-" * 40)
    print("预测标签")
    print("实际标签", end="")
    for class_name in all_classes:
        print(f"\t{class_name}", end="")
    print()
    
    for i, true_class in enumerate(all_classes):
        print(f"{true_class}", end="")
        for j, pred_class in enumerate(all_classes):
            print(f"\t{cm[i, j]}", end="")
        print()
    
    return results_df

if __name__ == "__main__":
    # 示例用法
    show_results()
