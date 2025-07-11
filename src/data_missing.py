import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] - %(message)s'
)

class MissingDataHandler:
    def __init__(self):
        self.output_dir = Path('Dataset/processed')
        with open('config/feature.json', 'r', encoding='utf-8') as f:
            feature_json = json.load(f)
            self.features_config = feature_json['features']  # 获取features字段
        self.train_data = None
        
    def load_normalized_data(self):
        """
        加载归一化后的数据
        """
        data_path = os.path.join(self.output_dir, 'normalized_train.csv')
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"归一化后的数据文件不存在: {data_path}")
        
        self.train_data = pd.read_csv(data_path)
        print(f"加载数据: {data_path}, 形状: {self.train_data.shape}")
        
    def analyze_missing_values(self):
        """
        分析每个特征的缺失值情况
        """
        print("\n缺失值分析:")
        missing_stats = {}
        
        for feature_name, feature_info in self.features_config.items():
            if not feature_info.get('iskeep', True):
                continue
                
            if feature_name not in self.train_data.columns:
                print(f"警告: 特征 {feature_name} 在数据集中未找到")
                continue
                
            missing_count = self.train_data[feature_name].isna().sum()
            total_count = len(self.train_data)
            missing_percentage = (missing_count / total_count) * 100
            
            missing_stats[feature_name] = {
                'missing_count': missing_count,
                'total_count': total_count,
                'missing_percentage': missing_percentage
            }
            
            if feature_info.get('missing', False):
                print(f"{feature_name}: {missing_count} 缺失值 ({missing_percentage:.2f}%)")
            
        return missing_stats
        
    def create_complete_datasets(self):
        """
        为每个特征创建无缺失值的数据集
        """
        os.makedirs(os.path.join(self.output_dir, 'complete_features'), exist_ok=True)
        
        for feature_name, feature_info in self.features_config.items():
            if not feature_info.get('iskeep', True):
                continue
                
            if feature_name not in self.train_data.columns:
                continue
                
            # 获取该特征无缺失值的数据
            complete_data = self.train_data.dropna(subset=[feature_name])
            
            # 如果数据集大小没有变化，说明没有缺失值，跳过保存
            if len(complete_data) == len(self.train_data):
                continue
                
            # 保存数据集
            output_path = os.path.join(self.output_dir, 'complete_features', f'{feature_name}_complete.csv')
            complete_data.to_csv(output_path, index=False)
            print(f"保存完整数据集: {output_path}, 形状: {complete_data.shape}")
            
    def create_all_complete_dataset(self):
        """
        创建所有特征都无缺失值的数据集
        """
        # 获取需要保留的特征列表
        kept_features = [
            feature_name for feature_name, feature_info in self.features_config.items()
            if feature_info.get('iskeep', True) and feature_name in self.train_data.columns
        ]
        
        # 删除所有选定特征中有缺失值的行
        complete_data = self.train_data.dropna(subset=kept_features)
        
        # 保存完整数据集
        output_path = os.path.join(self.output_dir, 'all_features_complete.csv')
        complete_data.to_csv(output_path, index=False)
        print(f"\n保存所有特征完整数据集: {output_path}")
        print(f"原始数据集大小: {self.train_data.shape}")
        print(f"完整数据集大小: {complete_data.shape}")
        print(f"删除的行数: {len(self.train_data) - len(complete_data)}")

def main():
    handler = MissingDataHandler()
    handler.load_normalized_data()
    handler.analyze_missing_values()
    handler.create_complete_datasets()
    handler.create_all_complete_dataset()

if __name__ == '__main__':
    main()
