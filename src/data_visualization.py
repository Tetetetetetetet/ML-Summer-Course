# -*- coding: utf-8 -*-
"""
糖尿病数据特征可视化分析
"""
import pandas as pd
from myutils import read_jsonl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import json
warnings.filterwarnings('ignore')

# 设置中文字体和样式
# plt.rcParams['font.sans-serif'] = ['STHeiti','Arial Unicode MS', 'SimHei', 'DejaVu Sans']
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['STHeiti']
plt.rcParams['axes.unicode_minus'] = False

class DataVisualizer:
    def __init__(self):
        self.dataset_path = Path(__file__).parent.parent / "Dataset"
        self.train_data = None
        self.test_data = None
        self.feature_file = Path(__file__).parent.parent / "feature.json"
        self.feature_json = read_jsonl(str(self.feature_file))
        
    def load_data(self):
        """加载数据"""
        print("正在加载数据...")
        self.train_data = pd.read_csv(self.dataset_path / "diabetic_data_training.csv")
        self.test_data = pd.read_csv(self.dataset_path / "diabetic_data_test.csv")
        print(f"训练集: {self.train_data.shape}, 测试集: {self.test_data.shape}")
        
    def get_demographic_features(self):
        """从feature.json中获取demographic特征"""
        demographic_features = []
        for feature_name, feature_info in self.feature_json['features'].items():
            if feature_info['category'] == 'demographic':
                demographic_features.append(feature_name)
        return demographic_features
        
    def process_demographic_features(self):
        """处理demographic特征，包括体重缺失值处理"""
        print("\n=== 处理Demographic特征 ===")
        
        demographic_features = self.get_demographic_features()
        print(f"Demographic特征: {demographic_features}")
        
        # 处理体重缺失值：将'?'视为一类，其他视为另一类
        if 'weight' in self.train_data.columns:
            # 创建新的weight_processed列
            self.train_data['weight_processed'] = self.train_data['weight'].apply(
                lambda x: 'Missing' if x == '?' else 'Available'
            )
            self.test_data['weight_processed'] = self.test_data['weight'].apply(
                lambda x: 'Missing' if x == '?' else 'Available'
            )
            print("体重缺失值处理完成")
            
        return demographic_features
        
    def visualize_demographic_features_histogram(self):
        """为demographic特征创建直方图，标注数量和比例"""
        print("\n=== Demographic特征直方图可视化 ===")
        
        demographic_features = self.get_demographic_features()
        
        # 创建子图，增加高度以避免重叠
        fig, axes = plt.subplots(2, 2, figsize=(15, 14))
        fig.suptitle('Demographic特征分布直方图', fontsize=16, fontweight='bold', y=0.95)
        
        for i, feature in enumerate(demographic_features):
            row, col = i // 2, i % 2
            
            if feature == 'weight':
                # 使用处理后的weight_processed
                feature_data = self.train_data['weight_processed']
            else:
                feature_data = self.train_data[feature]
            
            # 计算统计信息
            value_counts = feature_data.value_counts()
            total_count = len(feature_data)
            
            # 创建直方图
            bars = axes[row, col].bar(range(len(value_counts)), value_counts.values, 
                                    color='skyblue', alpha=0.7, edgecolor='black')
            axes[row, col].set_title(f'{feature} 分布', pad=20)
            axes[row, col].set_ylabel('数量')
            axes[row, col].set_xticks(range(len(value_counts)))
            axes[row, col].set_xticklabels(value_counts.index, rotation=45, ha='right')
            
            # 添加数量和比例标签
            for j, (value, count) in enumerate(value_counts.items()):
                percentage = (count / total_count) * 100
                axes[row, col].text(j, count + max(value_counts.values) * 0.01, 
                                  f'{count}\n({percentage:.1f}%)', 
                                  ha='center', va='bottom', fontsize=9)
            
            # 添加总数信息
            axes[row, col].text(0.02, 0.98, f'总数: {total_count}', 
                              transform=axes[row, col].transAxes, 
                              verticalalignment='top', fontsize=10,
                              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 调整布局，增加底部空间
        plt.tight_layout(pad=3.0)
        plt.subplots_adjust(bottom=0.15)  # 增加底部空间
        plt.savefig('src/visualizations/demographic_features_histogram.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_label_encoding(self):
        """为demographic特征创建标签编码（0,1,2,...序列）"""
        print("\n=== 创建Demographic特征标签编码 ===")
        
        demographic_features = self.get_demographic_features()
        label_encoding_data = {}
        
        for feature in demographic_features:
            if feature == 'weight':
                # 使用处理后的weight_processed
                feature_data = self.train_data['weight_processed']
            else:
                feature_data = self.train_data[feature]
            
            # 获取唯一值
            unique_values = feature_data.unique()
            print(f"{feature} 唯一值: {unique_values}")
            
            # 存储编码信息
            label_encoding_data[feature] = {
                'unique_values': unique_values.tolist(),
                'encoding_mapping': {}
            }
            
            # 创建标签编码映射（0,1,2,...）
            if feature == 'age':
                # 特殊处理age特征，让[0-10)编码为0
                sorted_values = sorted(unique_values)
                # 将[0-10)移到第一位
                if '[0-10)' in sorted_values:
                    sorted_values.remove('[0-10)')
                    sorted_values.insert(0, '[0-10)')
                
                for i, value in enumerate(sorted_values):
                    label_encoding_data[feature]['encoding_mapping'][str(value)] = i
            else:
                # 其他特征按原有顺序编码
                for i, value in enumerate(unique_values):
                    label_encoding_data[feature]['encoding_mapping'][str(value)] = i
            
            print(f"{feature} 标签编码完成，编码范围: 0-{len(unique_values)-1}")
        
        return label_encoding_data
        
    def update_feature_json(self, label_encoding_data):
        """直接以name为key写入features，并且不包含one-hot编码，只保留label_encoding和其它原有字段。"""
        print("\n=== 更新feature.json文件 ===")
        
        # 读取原始feature.json
        feature_data = self.feature_json
        
        # 构建新的features字典
        new_features = {}
        # 兼容旧结构（列表）和新结构（字典）
        features_iter = feature_data['features'].items() if isinstance(feature_data['features'], dict) else ((f['name'], f) for f in feature_data['features'])
        for feature_name, feature_info in features_iter:
            # 构建新特征字典（不包含one-hot编码）
            new_feature = {
                'feature_id': feature_info['feature_id'],
                'category': feature_info['category'],
                'type': feature_info['type'],
                'description': feature_info['description'],
                'visualization': feature_info['visualization']
            }
            # 如果有label_encoding，更新为最新的
            if feature_name in label_encoding_data:
                new_feature['label_encoding'] = label_encoding_data[feature_name]
            elif 'label_encoding' in feature_info:
                new_feature['label_encoding'] = feature_info['label_encoding']
            # 以name为key
            new_features[feature_name] = new_feature
        
        # 构建新结构
        new_data = {
            'dataset_name': feature_data['dataset_name'],
            'target_feature': feature_data['target_feature'],
            'features': new_features
        }
        
        # 保存新feature.json
        with open(self.feature_file, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, ensure_ascii=False, indent=2)
        print("feature.json文件已更新为以name为key的结构，并删除了one-hot编码，只保留label_encoding")
        
    def process_and_visualize_demographic(self):
        """处理demographic特征并创建可视化"""
        print("开始处理Demographic特征...")
        
        # 加载数据
        self.load_data()
        
        # 处理demographic特征
        demographic_features = self.process_demographic_features()
        
        # 创建直方图可视化
        self.visualize_demographic_features_histogram()
        
        # 创建标签编码
        label_encoding_data = self.create_label_encoding()
        
        # 更新feature.json
        self.update_feature_json(label_encoding_data)
        
        print("\nDemographic特征处理完成！")
        print(f"处理的特征: {demographic_features}")
        print("可视化图片保存在 src/visualizations/demographic_features_histogram.png")
        print("标签编码信息已更新到 feature.json")
        
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
        
        # 处理demographic特征
        self.process_and_visualize_demographic()
        
        print("\n所有可视化已完成！图片保存在 src/visualizations/ 目录中")

if __name__ == "__main__":
    visualizer = DataVisualizer()
    visualizer.create_all_visualizations() 