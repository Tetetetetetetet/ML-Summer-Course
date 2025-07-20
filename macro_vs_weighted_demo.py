#!/usr/bin/env python3
"""
演示 Macro Average vs Weighted Average 的差异
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report
import matplotlib.pyplot as plt

def demonstrate_difference():
    """
    演示macro avg和weighted avg的差异
    """
    print("========== Macro vs Weighted Average 差异演示 ==========")
    
    # 模拟您的数据分布
    n_samples = 90105
    class_distribution = {
        0: 47871,  # 53.2%
        1: 10245,  # 11.4%
        2: 31989   # 35.4%
    }
    
    print(f"数据分布:")
    for class_id, count in class_distribution.items():
        percentage = count / n_samples * 100
        print(f"  类别{class_id}: {count} 样本 ({percentage:.1f}%)")
    
    # 模拟不同场景的F1分数
    scenarios = {
        "场景1: 多数类表现好，少数类表现差": {
            "f1_scores": {0: 0.87, 1: 0.42, 2: 0.72},
            "description": "典型的不平衡数据情况"
        },
        "场景2: 各类别表现均衡": {
            "f1_scores": {0: 0.75, 1: 0.73, 2: 0.74},
            "description": "各类别表现相近"
        },
        "场景3: 少数类表现好，多数类表现差": {
            "f1_scores": {0: 0.45, 1: 0.85, 2: 0.50},
            "description": "少数类表现突出"
        }
    }
    
    results = []
    
    for scenario_name, scenario in scenarios.items():
        print(f"\n{scenario_name}")
        print(f"描述: {scenario['description']}")
        
        f1_scores = scenario['f1_scores']
        
        # 计算macro avg
        macro_f1 = np.mean(list(f1_scores.values()))
        
        # 计算weighted avg
        weighted_f1 = sum(count * f1_scores[class_id] for class_id, count in class_distribution.items()) / n_samples
        
        print(f"各类别F1分数:")
        for class_id in [0, 1, 2]:
            print(f"  类别{class_id}: {f1_scores[class_id]:.3f}")
        
        print(f"Macro Average F1: {macro_f1:.3f}")
        print(f"Weighted Average F1: {weighted_f1:.3f}")
        print(f"差异: {weighted_f1 - macro_f1:.3f}")
        
        results.append({
            'scenario': scenario_name,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'difference': weighted_f1 - macro_f1
        })
    
    return results

def analyze_medical_importance():
    """
    分析医疗场景中的重要性
    """
    print("\n========== 医疗场景重要性分析 ==========")
    
    medical_importance = {
        0: {"name": "正常出院", "importance": "中等", "cost_of_error": "中等"},
        1: {"name": "紧急情况", "importance": "高", "cost_of_error": "很高"},
        2: {"name": "延迟出院", "importance": "中等", "cost_of_error": "中等"}
    }
    
    print("各类别的医疗重要性:")
    for class_id, info in medical_importance.items():
        print(f"  类别{class_id} ({info['name']}): 重要性={info['importance']}, 错误代价={info['cost_of_error']}")
    
    print("\n推荐指标选择:")
    print("1. 如果关注整体预测准确性 → 选择 Weighted Average")
    print("2. 如果每个类别都同样重要 → 选择 Macro Average")
    print("3. 如果少数类(紧急情况)更重要 → 选择 Macro Average")
    print("4. 如果关注业务影响 → 选择 Weighted Average")

def create_visualization():
    """
    创建可视化图表
    """
    # 数据分布饼图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 饼图：数据分布
    labels = ['类别0\n(正常出院)', '类别1\n(紧急情况)', '类别2\n(延迟出院)']
    sizes = [47871, 10245, 31989]
    colors = ['lightblue', 'lightcoral', 'lightgreen']
    
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('数据分布')
    
    # 柱状图：不同场景的F1分数对比
    scenarios = ['场景1\n(多数类好)', '场景2\n(均衡)', '场景3\n(少数类好)']
    macro_scores = [0.67, 0.74, 0.60]
    weighted_scores = [0.77, 0.74, 0.48]
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    ax2.bar(x - width/2, macro_scores, width, label='Macro Average', color='skyblue')
    ax2.bar(x + width/2, weighted_scores, width, label='Weighted Average', color='lightcoral')
    
    ax2.set_xlabel('场景')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('不同场景下的F1分数对比')
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('macro_vs_weighted_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    results = demonstrate_difference()
    analyze_medical_importance()
    create_visualization()
    
    print("\n========== 总结 ==========")
    print("1. Macro Average 更公平地评估每个类别")
    print("2. Weighted Average 更反映整体业务表现")
    print("3. 在医疗场景中，建议根据业务需求选择:")
    print("   - 如果紧急情况预测错误代价高 → 选择 Macro")
    print("   - 如果关注整体预测准确性 → 选择 Weighted") 