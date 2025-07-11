import pandas as pd
from myutils import read_jsonl,write_jsonl
import numpy as np
import json
import os
from pathlib import Path
import logging
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] - %(message)s'
)

class MissingDataHandler:
    def __init__(self):
        self.output_dir = Path('Dataset/processed')
        self.feature_json = read_jsonl('config/feature.json')
        self.features_config = self.feature_json['features']  # 获取features字段
        self.train_data = None
        
    def load_normalized_data(self):
        """
        加载归一化后的数据
        """
        data_path = os.path.join(self.output_dir, 'normalized_train.csv')
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"归一化后的数据文件不存在: {data_path}")
        
        self.train_data = pd.read_csv(data_path)
        logging.info(f"加载数据: {data_path}, 形状: {self.train_data.shape}")
        
    def analyze_missing_values(self):
        """
        分析每个特征的缺失值情况, 按照缺失率分为三类:
        - mid: 缺失率在0.1-0.8之间
        - high: 缺失率大于0.8
        - low: 缺失率小于0.1
        """
        logging.info("\n缺失值分析:")
        missing_stats = {}
        
        # 初始化missing_type字段
        self.feature_json['missing_type'] = {'mid':[],'low':[],'high':[]}
        
        for feature_name, feature_info in self.features_config.items():
            if not feature_info.get('iskeep', True) or feature_info['missing']==False:
                continue
                
            if feature_name not in self.train_data.columns:
                logging.warning(f"特征 {feature_name} 在数据集中未找到")
                continue
                
            missing_count = self.train_data[feature_name].isna().sum()
            total_count = len(self.train_data)
            missing_percentage = (missing_count / total_count) * 100
            
            if feature_info.get('missing', False):
                logging.info(f"{feature_name}: {missing_count} 缺失值 ({missing_percentage:.2f}%)")
            
            # 根据缺失率分类
            if missing_percentage > 10 and missing_percentage < 80:
                self.feature_json['missing_type']['mid'].append(feature_name)
            elif missing_percentage > 80:
                self.feature_json['missing_type']['high'].append(feature_name)
            else:
                self.feature_json['missing_type']['low'].append(feature_name)
            
        write_jsonl(self.feature_json,'config/feature.json')
        return missing_stats
        
    def drop_low_missing_dataset(self):
        """
        1. 只保留低缺失率特征
        2. 删除这些特征中含有缺失值的行
        返回完整的数据集（无缺失值）
        """
        # 获取低缺失率特征
        low_missing_features = self.feature_json['missing_type']['low']
        
        # 只保留低缺失率特征
        data = self.train_data.dropna(subset=low_missing_features)
        data = data.drop(columns=self.feature_json['missing_type']['mid']+self.feature_json['missing_type']['high'])
        
        # 删除含有缺失值的行
        complete_data = data.dropna()
        try:
            assert len(complete_data) == len(data)
        except AssertionError:
            logging.error(f"删除低缺失率特征的行后，数据集大小不一致: {len(data)} != {len(complete_data)}")
            pdb.set_trace()
            raise AssertionError
        
        # 记录删除的行数
        rows_dropped = len(data) - len(complete_data)
        logging.info(f"保留的特征: {low_missing_features}")
        logging.info(f"删除含缺失值的行: {rows_dropped} 行 ({rows_dropped/len(data)*100:.2f}%)")
        logging.info(f"剩余数据集大小: {len(complete_data)} 行")
        
        return complete_data
    
    def drop_mid_missing_dataset(self):
        """
        删除中缺失率特征的行
        """
        mid_missing_features = self.feature_json['missing_type']['mid']
        return self.train_data.dropna(subset=mid_missing_features)
    
    def ppca(self, data: pd.DataFrame):
        '''
        对于数据集data做主成分分析
        args:
            - data: pd.DataFrame, 完整的数据集（无缺失值）
        '''
        # 创建可视化输出目录
        vis_dir = self.output_dir / 'ppca_analysis'
        os.makedirs(vis_dir, exist_ok=True)
        
        # 数据标准化
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        scaled_df = pd.DataFrame(scaled_data, columns=data.columns)
        
        # 执行PCA
        pca = PCA()
        pca_result = pca.fit_transform(scaled_data)
        
        # 计算解释方差比
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
        
        # 绘制碎石图
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, 'bo-')
        plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, 'ro-')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Scree Plot')
        plt.legend(['Individual', 'Cumulative'])
        plt.grid(True)
        plt.savefig(vis_dir / 'scree_plot.png')
        plt.close()
        
        # 计算特征对主成分的贡献
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(pca.n_components_)],
            index=data.columns
        )
        
        # 绘制特征贡献热力图
        plt.figure(figsize=(12, 8))
        sns.heatmap(loadings.iloc[:, :5], annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Loadings (First 5 PCs)')
        plt.tight_layout()
        plt.savefig(vis_dir / 'feature_loadings.png')
        plt.close()
        
        # 保存分析结果
        analysis_results = {
            'explained_variance_ratio': explained_variance_ratio.tolist(),
            'cumulative_variance_ratio': cumulative_variance_ratio.tolist(),
            'feature_loadings': loadings.to_dict(),
            'data_shape': {
                'samples': len(data),
                'features': len(data.columns)
            }
        }
        
        with open(vis_dir / 'pca_analysis.json', 'w') as f:
            json.dump(analysis_results, f, indent=4)
        
        logging.info(f"PPCA分析完成，结果保存在: {vis_dir}")
        return pca, loadings

    def process_low_missing_features(self):
        """
        处理低缺失率特征
        """
        low_missing_features = self.feature_json['missing_type']['low']
        new_data = self.drop_low_missing_dataset()
        logging.info(f"删除低缺失率特征的行: {len(self.train_data) - len(new_data)} rows dropped")
        return new_data

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
            logging.info(f"保存完整数据集: {output_path}, 形状: {complete_data.shape}")
            
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
        logging.info(f"\n保存所有特征完整数据集: {output_path}")
        logging.info(f"原始数据集大小: {self.train_data.shape}")
        logging.info(f"完整数据集大小: {complete_data.shape}")
        logging.info(f"删除的行数: {len(self.train_data) - len(complete_data)}")

def main():
    handler = MissingDataHandler()
    handler.load_normalized_data()
    handler.analyze_missing_values()
    low_complete_data = handler.process_low_missing_features()
    handler.ppca(low_complete_data)
    return
    mid_complete_data = handler.process_mid_missing_features()
    handler.create_complete_datasets()
    handler.create_all_complete_dataset()

if __name__ == '__main__':
    main()
