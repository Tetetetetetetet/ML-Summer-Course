import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 假设我们有以下数值型变量
numerical_columns = [
    'time_in_hospital',      # 住院时间
    'num_lab_procedures',    # 实验室检查次数
    'num_procedures',        # 手术次数
    'num_medications',       # 用药数量
    'number_outpatient',     # 门诊次数
    'number_emergency',      # 急诊次数
    'number_inpatient',      # 住院次数
    'number_diagnoses'       # 诊断数量
]

def analyze_numerical_variable(df, column_name, target_column='readmitted'):
    """
    对数值型变量进行完整分析
    """
    print(f"=== {column_name} 变量分析 ===")
    
    # 1. 基本统计量
    print("\n1. 基本统计量:")
    stats_summary = df[column_name].describe()
    print(stats_summary)
    
    # 计算偏度和峰度
    skewness = df[column_name].skew()
    kurtosis = df[column_name].kurtosis()
    print(f"偏度: {skewness:.4f}")
    print(f"峰度: {kurtosis:.4f}")
    
    # 2. 缺失值统计
    missing_count = df[column_name].isnull().sum()
    missing_pct = (missing_count / len(df)) * 100
    print(f"\n2. 缺失值统计:")
    print(f"缺失数量: {missing_count}")
    print(f"缺失比例: {missing_pct:.2f}%")
    
    # 3. 异常值检测（使用IQR方法）
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column_name] < lower_bound) | (df[column_name] > upper_bound)]
    outlier_count = len(outliers)
    outlier_pct = (outlier_count / len(df)) * 100
    
    print(f"\n3. 异常值检测 (IQR方法):")
    print(f"异常值数量: {outlier_count}")
    print(f"异常值比例: {outlier_pct:.2f}%")
    print(f"异常值范围: < {lower_bound:.2f} 或 > {upper_bound:.2f}")
    
    # 4. 按目标变量分组的统计
    print(f"\n4. 按再入院情况分组的统计:")
    grouped_stats = df.groupby(target_column)[column_name].agg([
        'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'
    ]).round(2)
    print(grouped_stats)
    
    # 5. 相关性分析
    correlation = df[column_name].corr(df[target_column])
    print(f"\n5. 与目标变量的相关性:")
    print(f"皮尔逊相关系数: {correlation:.4f}")
    
    # 6. 可视化
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{column_name} 变量分析', fontsize=16)
    
    # 直方图
    axes[0, 0].hist(df[column_name].dropna(), bins=30, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('分布直方图')
    axes[0, 0].set_xlabel(column_name)
    axes[0, 0].set_ylabel('频数')
    
    # 箱线图
    axes[0, 1].boxplot(df[column_name].dropna())
    axes[0, 1].set_title('箱线图')
    axes[0, 1].set_ylabel(column_name)
    
    # 按目标变量分组的箱线图
    df.boxplot(column=column_name, by=target_column, ax=axes[1, 0])
    axes[1, 0].set_title(f'按{target_column}分组的箱线图')
    axes[1, 0].set_xlabel(target_column)
    axes[1, 0].set_ylabel(column_name)
    
    # 散点图
    axes[1, 1].scatter(df[column_name], df[target_column], alpha=0.5)
    axes[1, 1].set_xlabel(column_name)
    axes[1, 1].set_ylabel(target_column)
    axes[1, 1].set_title(f'{column_name} vs {target_column}')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'stats_summary': stats_summary,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'missing_count': missing_count,
        'missing_pct': missing_pct,
        'outlier_count': outlier_count,
        'outlier_pct': outlier_pct,
        'correlation': correlation,
        'grouped_stats': grouped_stats
    }

def create_numerical_summary_table(df, numerical_columns, target_column='readmitted'):
    """
    创建数值型变量的汇总表
    """
    summary_data = []
    
    for col in numerical_columns:
        if col in df.columns:
            # 基本统计
            stats = df[col].describe()
            missing_count = df[col].isnull().sum()
            missing_pct = (missing_count / len(df)) * 100
            skewness = df[col].skew()
            kurtosis = df[col].kurtosis()
            correlation = df[col].corr(df[target_column])
            
            # 按目标变量分组的均值
            grouped_means = df.groupby(target_column)[col].mean()
            
            summary_data.append({
                '变量名': col,
                '样本数': stats['count'],
                '均值': round(stats['mean'], 2),
                '标准差': round(stats['std'], 2),
                '最小值': stats['min'],
                '25%分位数': round(stats['25%'], 2),
                '中位数': round(stats['50%'], 2),
                '75%分位数': round(stats['75%'], 2),
                '最大值': stats['max'],
                '偏度': round(skewness, 3),
                '峰度': round(kurtosis, 3),
                '缺失数': missing_count,
                '缺失比例(%)': round(missing_pct, 2),
                '与目标变量相关系数': round(correlation, 4),
                '未再入院均值': round(grouped_means.get(2, 0), 2),
                '>30天再入院均值': round(grouped_means.get(1, 0), 2),
                '<30天再入院均值': round(grouped_means.get(0, 0), 2)
            })
    
    summary_df = pd.DataFrame(summary_data)
    return summary_df

# 使用示例
if __name__ == "__main__":
    # 假设df是您的数据集
    # df = pd.read_csv('your_data.csv')
    
    # 分析单个数值变量
    # results = analyze_numerical_variable(df, 'time_in_hospital')
    
    # 创建汇总表
    # summary_table = create_numerical_summary_table(df, numerical_columns)
    # print(summary_table.to_string(index=False))
    
    print("数值型变量分析代码已准备就绪！") 