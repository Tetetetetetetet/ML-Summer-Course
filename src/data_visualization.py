# -*- coding: utf-8 -*-
"""
糖尿病数据特征可视化分析
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

class DataVisualizer:
    def __init__(self):
        self.dataset_path = Path(__file__).parent.parent / "Dataset"
        self.train_data = None
        self.test_data = None
        
    def load_data(self):
        """加载数据"""
        print("正在加载数据...")
        self.train_data = pd.read_csv(self.dataset_path / "diabetic_data_training.csv")
        self.test_data = pd.read_csv(self.dataset_path / "diabetic_data_test.csv")
        print(f"训练集: {self.train_data.shape}, 测试集: {self.test_data.shape}")
        
    def visualize_demographic_features(self):
        """可视化人口统计学特征"""
        print("\n=== 人口统计学特征可视化 ===")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('人口统计学特征分布', fontsize=16, fontweight='bold')
        
        # 种族分布
        race_counts = self.train_data['race'].value_counts()
        axes[0, 0].pie(race_counts.values, labels=race_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('种族分布')
        
        # 性别分布
        gender_counts = self.train_data['gender'].value_counts()
        axes[0, 1].bar(gender_counts.index, gender_counts.values, color=['skyblue', 'lightcoral'])
        axes[0, 1].set_title('性别分布')
        axes[0, 1].set_ylabel('数量')
        
        # 年龄分布
        age_counts = self.train_data['age'].value_counts().sort_index()
        axes[1, 0].bar(range(len(age_counts)), age_counts.values, color='lightgreen')
        axes[1, 0].set_title('年龄分布')
        axes[1, 0].set_xlabel('年龄组')
        axes[1, 0].set_ylabel('数量')
        axes[1, 0].set_xticks(range(len(age_counts)))
        axes[1, 0].set_xticklabels(age_counts.index, rotation=45)
        
        # 体重分布（缺失值处理）
        weight_counts = self.train_data['weight'].value_counts()
        axes[1, 1].pie(weight_counts.values, labels=weight_counts.index, autopct='%1.1f%%')
        axes[1, 1].set_title('体重分布')
        
        plt.tight_layout()
        plt.savefig('src/visualizations/demographic_features.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def visualize_numeric_features(self):
        """可视化数值型特征"""
        print("\n=== 数值型特征可视化 ===")
        
        numeric_features = ['time_in_hospital', 'num_lab_procedures', 'num_procedures',
                           'num_medications', 'number_outpatient', 'number_emergency',
                           'number_inpatient', 'number_diagnoses']
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('数值型特征分布', fontsize=16, fontweight='bold')
        
        for i, feature in enumerate(numeric_features):
            row, col = i // 4, i % 4
            
            # 直方图
            axes[row, col].hist(self.train_data[feature], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
            axes[row, col].set_title(f'{feature} 分布')
            axes[row, col].set_xlabel(feature)
            axes[row, col].set_ylabel('频次')
            
            # 添加统计信息
            mean_val = self.train_data[feature].mean()
            median_val = self.train_data[feature].median()
            axes[row, col].axvline(mean_val, color='red', linestyle='--', label=f'均值: {mean_val:.2f}')
            axes[row, col].axvline(median_val, color='green', linestyle='--', label=f'中位数: {median_val:.2f}')
            axes[row, col].legend()
        
        plt.tight_layout()
        plt.savefig('src/visualizations/numeric_features.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def visualize_categorical_features(self):
        """可视化类别型特征"""
        print("\n=== 类别型特征可视化 ===")
        
        categorical_features = ['admission_type_id', 'discharge_disposition_id', 'admission_source_id',
                               'payer_code', 'medical_specialty', 'max_glu_serum', 'A1Cresult', 'change', 'diabetesMed']
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('类别型特征分布', fontsize=16, fontweight='bold')
        
        for i, feature in enumerate(categorical_features):
            row, col = i // 3, i % 3
            
            # 获取前10个最常见的值
            value_counts = self.train_data[feature].value_counts().head(10)
            
            # 水平条形图
            axes[row, col].barh(range(len(value_counts)), value_counts.values, color='lightcoral')
            axes[row, col].set_yticks(range(len(value_counts)))
            axes[row, col].set_yticklabels(value_counts.index)
            axes[row, col].set_title(f'{feature} 分布 (Top 10)')
            axes[row, col].set_xlabel('数量')
            
            # 添加数值标签
            for j, v in enumerate(value_counts.values):
                axes[row, col].text(v, j, str(v), va='center')
        
        plt.tight_layout()
        plt.savefig('src/visualizations/categorical_features.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def visualize_medication_features(self):
        """可视化药物特征"""
        print("\n=== 药物特征可视化 ===")
        
        medication_features = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
                              'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
                              'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
                              'miglitol', 'troglitazone', 'tolazamide', 'examide',
                              'citoglipton', 'insulin', 'glyburide-metformin',
                              'glipizide-metformin', 'glimepiride-pioglitazone',
                              'metformin-rosiglitazone', 'metformin-pioglitazone']
        
        # 计算每种药物的使用情况
        medication_usage = {}
        for med in medication_features:
            if med in self.train_data.columns:
                usage_counts = self.train_data[med].value_counts()
                medication_usage[med] = usage_counts
        
        # 创建药物使用热力图
        fig, axes = plt.subplots(1, 1, figsize=(15, 8))
        
        # 准备热力图数据
        usage_matrix = []
        med_names = []
        for med, counts in medication_usage.items():
            if len(counts) > 0:
                usage_matrix.append(counts.values)
                med_names.append(med)
        
        if usage_matrix:
            usage_df = pd.DataFrame(usage_matrix, index=med_names, columns=['No', 'Steady', 'Up', 'Down'])
            
            sns.heatmap(usage_df, annot=True, fmt='.0f', cmap='YlOrRd', ax=axes)
            axes.set_title('药物使用情况热力图')
            axes.set_xlabel('使用状态')
            axes.set_ylabel('药物名称')
        
        plt.tight_layout()
        plt.savefig('src/visualizations/medication_features.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def visualize_diagnosis_features(self):
        """可视化诊断特征"""
        print("\n=== 诊断特征可视化 ===")
        
        diagnosis_features = ['diag_1', 'diag_2', 'diag_3']
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle('诊断特征分布', fontsize=16, fontweight='bold')
        
        for i, feature in enumerate(diagnosis_features):
            # 获取前15个最常见的诊断
            diag_counts = self.train_data[feature].value_counts().head(15)
            
            axes[i].barh(range(len(diag_counts)), diag_counts.values, color='lightblue')
            axes[i].set_yticks(range(len(diag_counts)))
            axes[i].set_yticklabels(diag_counts.index)
            axes[i].set_title(f'{feature} 分布 (Top 15)')
            axes[i].set_xlabel('数量')
            
            # 添加数值标签
            for j, v in enumerate(diag_counts.values):
                axes[i].text(v, j, str(v), va='center')
        
        plt.tight_layout()
        plt.savefig('src/visualizations/diagnosis_features.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def visualize_target_feature(self):
        """可视化目标特征"""
        print("\n=== 目标特征可视化 ===")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('目标特征 - 复诊情况', fontsize=16, fontweight='bold')
        
        # 复诊情况分布
        readmitted_counts = self.train_data['readmitted'].value_counts()
        axes[0].pie(readmitted_counts.values, labels=readmitted_counts.index, autopct='%1.1f%%', 
                   colors=['lightgreen', 'lightcoral', 'lightblue'])
        axes[0].set_title('复诊情况分布')
        
        # 复诊情况条形图
        axes[1].bar(readmitted_counts.index, readmitted_counts.values, 
                   color=['lightgreen', 'lightcoral', 'lightblue'])
        axes[1].set_title('复诊情况数量')
        axes[1].set_ylabel('数量')
        
        # 添加数值标签
        for i, v in enumerate(readmitted_counts.values):
            axes[1].text(i, v, str(v), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('src/visualizations/target_feature.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def visualize_correlations(self):
        """可视化特征相关性"""
        print("\n=== 特征相关性可视化 ===")
        
        # 选择数值型特征进行相关性分析
        numeric_features = ['time_in_hospital', 'num_lab_procedures', 'num_procedures',
                           'num_medications', 'number_outpatient', 'number_emergency',
                           'number_inpatient', 'number_diagnoses']
        
        correlation_matrix = self.train_data[numeric_features].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5)
        plt.title('数值型特征相关性矩阵')
        plt.tight_layout()
        plt.savefig('src/visualizations/correlations.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def visualize_missing_values(self):
        """可视化缺失值情况"""
        print("\n=== 缺失值可视化 ===")
        
        # 计算缺失值
        missing_data = self.train_data.isnull().sum()
        missing_pct = (missing_data / len(self.train_data)) * 100
        
        # 创建缺失值数据框
        missing_df = pd.DataFrame({
            '特征': missing_data.index,
            '缺失数量': missing_data.values,
            '缺失比例(%)': missing_pct.values
        })
        
        # 只显示有缺失值的特征
        missing_df = missing_df[missing_df['缺失数量'] > 0].sort_values('缺失比例(%)', ascending=False)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('缺失值分析', fontsize=16, fontweight='bold')
        
        # 缺失值数量条形图
        axes[0].bar(range(len(missing_df)), missing_df['缺失数量'], color='red', alpha=0.7)
        axes[0].set_title('缺失值数量')
        axes[0].set_xlabel('特征')
        axes[0].set_ylabel('缺失数量')
        axes[0].set_xticks(range(len(missing_df)))
        axes[0].set_xticklabels(missing_df['特征'], rotation=45)
        
        # 缺失值比例条形图
        axes[1].bar(range(len(missing_df)), missing_df['缺失比例(%)'], color='orange', alpha=0.7)
        axes[1].set_title('缺失值比例')
        axes[1].set_xlabel('特征')
        axes[1].set_ylabel('缺失比例(%)')
        axes[1].set_xticks(range(len(missing_df)))
        axes[1].set_xticklabels(missing_df['特征'], rotation=45)
        
        plt.tight_layout()
        plt.savefig('src/visualizations/missing_values.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_all_visualizations(self):
        """创建所有可视化"""
        print("开始创建数据可视化...")
        
        # 创建保存目录
        save_dir = Path(__file__).parent / "visualizations"
        save_dir.mkdir(exist_ok=True)
        
        # 加载数据
        self.load_data()
        
        # 创建各种可视化
        self.visualize_demographic_features()
        # self.visualize_numeric_features()
        # self.visualize_categorical_features()
        # self.visualize_medication_features()
        # self.visualize_diagnosis_features()
        # self.visualize_target_feature()
        # self.visualize_correlations()
        # self.visualize_missing_values()
        
        print("\n所有可视化已完成！图片保存在 src/visualizations/ 目录中")

if __name__ == "__main__":
    visualizer = DataVisualizer()
    visualizer.create_all_visualizations() 