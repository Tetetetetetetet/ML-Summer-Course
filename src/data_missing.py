import pandas as pd
import pdb
from sklearn.impute import KNNImputer
from myutils import read_jsonl,write_jsonl
import numpy as np
import json
import os
from pathlib import Path
import logging
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)

class MissingDataHandler:
    def __init__(self):
        self.output_dir = Path('Dataset/processed/train_processed')
        self.feature_json = read_jsonl('config/feature.json')
        self.features_config = self.feature_json['features']  # 获取features字段
        self.train_data = None
        
    def load_data(self,mode:str='recoded'):
        """
        加载recoded后的数据
        """
        if mode=='recoded':
            data_path = os.path.join(self.output_dir, 'recoded_train.csv')
        elif mode=='normalized':
            data_path = os.path.join(self.output_dir, 'normalized_train.csv')
        else:
            raise ValueError(f"不支持的加载模式: {mode}")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"recoded后的数据文件不存在: {data_path}")
        self.train_data = pd.read_csv(data_path)
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
        logging.info("==========缺失值分析==========")
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
        
    def get_low_missing_complete_dataset(self):
        """
        1. 只保留低缺失率特征
        2. 删除这些特征中含有缺失值的行
        返回完整的数据集（无缺失值）
        """
        # 获取低缺失率特征
        logging.info("==========get_low_missing_complete_dataset==========")
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
        rows_dropped = len(self.train_data) - len(complete_data)
        logging.info(f"低缺失率特征: {low_missing_features}")
        logging.info(f"删除缺失[低]缺失率特征的行 & [中,高]缺失率特征: {rows_dropped} 行 ({rows_dropped/len(data)*100:.2f}%)")
        logging.info(f"剩余数据集大小: {len(complete_data)} 行")
        
        return complete_data
    
    def get_mid_missing_complete_dataset(self):
        """
        1. 只保留中缺失率特征
        2. 删除这些特征中含有缺失值的行
        3. 删除其他缺失特征
        """
        logging.info("==========get_mid_missing_complete_dataset==========")
        mid_missing_features = self.feature_json['missing_type']['mid']
        data = self.train_data.dropna(subset=mid_missing_features)
        data = data.drop(columns=self.feature_json['missing_type']['high']+self.feature_json['missing_type']['low'])
        complete_data = data.dropna()
        assert len(complete_data) == len(data)
        rows_dropped = len(self.train_data) - len(data)
        logging.info(f"中缺失率特征: {mid_missing_features}")
        logging.info(f"删除缺失[中]缺失率特征的行 & [低,高]缺失率特征: {rows_dropped} 行 ({rows_dropped/len(self.train_data)*100:.2f}%)")
        logging.info(f"剩余数据集大小: {len(data)} 行")
        return data
    
    def get_high_missing_complete_dataset(self):
        """
        1. 只保留高缺失率特征
        2. 删除这些特征中含有缺失值的行
        3. 删除其他缺失特征
        """
        logging.info("==========get_high_missing_complete_dataset==========")
        high_missing_features = self.feature_json['missing_type']['high']
        data = self.train_data.dropna(subset=high_missing_features)
        data = data.drop(columns=self.feature_json['missing_type']['mid']+self.feature_json['missing_type']['low'])
        complete_data = data.dropna()
        assert len(complete_data) == len(data)
        rows_dropped = len(self.train_data) - len(data)
        logging.info(f"高缺失率特征: {high_missing_features}")
        logging.info(f"删除缺失[高]缺失率特征的行 & [中,低]缺失率特征: {rows_dropped} 行 ({rows_dropped/len(self.train_data)*100:.2f}%)")
        logging.info(f"剩余数据集大小: {len(data)} 行")
        return data

    def pca_for_low_mid_missing_dataset(self):
        """
        对于低缺失率和中缺失率特征，进行主成分分析
        """
        low_complete_data = handler.process_low_missing_features()
        handler.pca(low_complete_data,'low_missing_complete')
        mid_complete_data = handler.get_mid_missing_complete_dataset()
        handler.pca(mid_complete_data,'mid_missing_complete')

    def pca(self, data: pd.DataFrame,exp_name:str='first_try'):
        '''
        对于数据集data做主成分分析
        args:
            - data: pd.DataFrame, 完整的数据集（无缺失值）
        '''
        logging.info("==========pca==========")
        # 创建可视化输出目录
        vis_dir = self.output_dir /'pca_analysis'/f'{exp_name}'
        os.makedirs(vis_dir, exist_ok=True)
        
        # 去掉label
        data = data.drop(columns=['readmitted'])
        
        # 数据预处理
        # 1. 分析特征类型
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        logging.info(f"数值型特征 ({len(numeric_cols)}):")
        for col in numeric_cols:
            unique_vals = data[col].nunique()
            val_range = data[col].describe()
            logging.info(f"  - {col}: {unique_vals} 个唯一值, 范围[{val_range['min']:.2f}, {val_range['max']:.2f}]")
        
        categorical_cols = data.select_dtypes(exclude=[np.number]).columns
        if len(categorical_cols) > 0:
            logging.info(f"非数值型特征 ({len(categorical_cols)}):")
            for col in categorical_cols:
                logging.info(f"  - {col}: {data[col].nunique()} 个唯一值")
        
        # 2. 检查每个数值特征的分布
        for col in numeric_cols:
            plt.figure(figsize=(10, 6))
            plt.subplot(2, 1, 1)
            plt.hist(data[col], bins=50)
            plt.title(f'Distribution of {col}')
            plt.subplot(2, 1, 2)
            plt.boxplot(data[col])
            plt.title(f'Boxplot of {col}')
            plt.tight_layout()
            plt.savefig(vis_dir / f'distribution_{col}.png')
            plt.close()
        
        # 3. 检查是否有应该被当作分类变量的数值特征
        potential_categorical = []
        for col in numeric_cols:
            if data[col].nunique() < 10:  # 如果唯一值少于10个，可能是分类变量
                potential_categorical.append(col)
                logging.warning(f"特征 '{col}' 可能是分类变量 (唯一值: {sorted(data[col].unique())})")
        
        # 4. 检查并处理极端值
        outlier_stats = {}
        cleaned_data = data.copy()
        
        for col in numeric_cols:
            if col in potential_categorical:
                continue  # 跳过可能的分类变量
                
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)][col]
            if len(outliers) > 0:
                outlier_stats[col] = {
                    'count': len(outliers),
                    'percentage': (len(outliers) / len(data)) * 100,
                    'min': float(outliers.min()),
                    'max': float(outliers.max()),
                    'normal_range': {
                        'lower': float(lower_bound),
                        'upper': float(upper_bound)
                    }
                }
                logging.warning(f"特征 '{col}' 有 {len(outliers)} 个极端值 ({(len(outliers)/len(data))*100:.2f}%)")
                logging.warning(f"  - 正常范围: [{lower_bound:.2f}, {upper_bound:.2f}]")
                logging.warning(f"  - 极端值范围: [{outliers.min():.2f}, {outliers.max():.2f}]")
                
                # 处理极端值：截断到上下界
                cleaned_data.loc[data[col] < lower_bound, col] = lower_bound
                cleaned_data.loc[data[col] > upper_bound, col] = upper_bound
        
        # 保存极端值统计
        with open(vis_dir / 'outlier_stats.json', 'w') as f:
            json.dump(outlier_stats, f, indent=4)
        
        # 5. 数据标准化
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cleaned_data[numeric_cols])
        scaled_df = pd.DataFrame(scaled_data, columns=numeric_cols)
        
        # 检查标准化后的数据
        logging.info("标准化后的数据统计:")
        for col in scaled_df.columns:
            stats = scaled_df[col].describe()
            logging.info(f"  - {col}: mean={stats['mean']:.2f}, std={stats['std']:.2f}")
        
        # 6. 验证标准化后的数据
        if np.isnan(scaled_data).any() or np.isinf(scaled_data).any():
            raise ValueError("标准化后的数据包含无效值(NaN或Inf)")
        
        # 7. 执行PCA
        pca = PCA()
        try:
            pca_result = pca.fit_transform(scaled_data)
        except Exception as e:
            logging.error(f"PCA计算失败: {e}")
            logging.error(f"数据统计: \n{scaled_df.describe()}")
            raise
        
        # 计算解释方差比
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
        
        # 保存数据统计信息
        stats = {
            'n_samples': len(data),
            'n_features': len(numeric_cols),
            'n_outliers_total': sum(s['count'] for s in outlier_stats.values()),
            'features': list(numeric_cols),
            'feature_stats': cleaned_data[numeric_cols].describe().to_dict(),
            'outlier_stats': outlier_stats,
            'potential_categorical': list(potential_categorical)
        }
        with open(vis_dir / 'data_stats.json', 'w') as f:
            json.dump(stats, f, indent=4)
        
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
            index=numeric_cols
        )
        
        # 绘制特征贡献热力图
        plt.figure(figsize=(12, 8))
        sns.heatmap(loadings.iloc[:, :20], annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Loadings (First 20 PCs)')
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
                'features': len(numeric_cols)
            }
        }
        
        with open(vis_dir / 'pca_analysis.json', 'w') as f:
            json.dump(analysis_results, f, indent=4)
        
        logging.info(f"PCA分析完成，结果保存在: {vis_dir}")
        return pca, loadings

    def drop_high_missing_dataset(self,data:pd.DataFrame):
        """
        删除高缺失率特征
        """
        logging.info("==========drop_high_missing_dataset==========")
        high_missing_features = self.feature_json['missing_type']['high']
        data = data.drop(columns=high_missing_features)
        logging.info(f"删除高缺失率特征: {high_missing_features}, left {len(data)} rows")
        return data

    def drop_mid_missing_dataset(self,data:pd.DataFrame):
        """
        删除中缺失率特征
        """
        logging.info("==========drop_mid_missing_dataset==========")
        mid_missing_features = self.feature_json['missing_type']['mid']
        data = data.drop(columns=mid_missing_features)
        logging.info(f"删除中缺失率特征: {mid_missing_features}, left {len(data)} rows")
        return data

    def drop_low_missing_dataset(self,data:pd.DataFrame):
        """
        删除低缺失率特征
        """
        logging.info("==========drop_low_missing_dataset==========")
        low_missing_features = self.feature_json['missing_type']['low']
        data = data.drop(columns=low_missing_features)
        logging.info(f"删除低缺失率特征: {low_missing_features}, left {len(data)} rows")
        return data

    def knn_impute(self,data:pd.DataFrame):
        """
        使用KNN填补缺失值
        """
        logging.info("==========knn_impute==========")
        columns = data.columns
        imputer = KNNImputer(n_neighbors=5)
        data = imputer.fit_transform(data[columns])
        data = pd.DataFrame(data, columns=columns)
        data.to_csv(self.output_dir/'knn_imputed_train.csv',index=False)
        return data

    def logistic_regression_impute(self, data: pd.DataFrame, target_feature: str = None):
        """
        使用逻辑回归填补缺失值
        对于每个有缺失值的特征，训练一个逻辑回归模型来预测缺失值
        
        Args:
            data: 包含缺失值的数据
            target_feature: 指定要填补的特征，如果为None则填补所有有缺失值的特征
        """
        logging.info("==========logistic_regression_impute==========")
        
        # 创建输出目录
        imputed_dir = os.path.join(self.output_dir, 'logistic_imputed')
        os.makedirs(imputed_dir, exist_ok=True)
        
        # 创建模型保存目录
        models_dir = os.path.join(imputed_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        # 获取有缺失值的特征
        missing_features = []
        for col in data.columns:
            if data[col].isna().sum() > 0:
                missing_features.append(col)
        
        if target_feature:
            if target_feature not in missing_features:
                logging.warning(f"特征 '{target_feature}' 没有缺失值")
                return data
            missing_features = [target_feature]
        
        logging.info(f"需要填补的特征: {missing_features}")
        
        # 保存模型信息的字典
        model_info = {
            'imputation_method': 'logistic_regression',
            'features_imputed': [],
            'models': {},
            'feature_stats': {},
            'imputation_stats': {}
        }
        
        imputed_data = data.copy()
        
        for feature in missing_features:
            logging.info(f"正在填补特征: {feature}")
            
            # 获取该特征的缺失值位置
            missing_mask = data[feature].isna()
            missing_count = missing_mask.sum()
            
            if missing_count == 0:
                continue
            
            # 获取非缺失值的数据作为训练集
            train_mask = ~missing_mask
            train_data = data[train_mask].copy()

            # 如果没有可用样本，跳过
            if train_data.shape[0] == 0:
                logging.warning(f"特征 {feature} 没有可用样本，跳过填补")
                continue
            
            # 准备特征和目标变量
            X_train = train_data.drop(columns=[feature])
            y_train = train_data[feature]
            
            # 对所有特征做缺失值填充
            for col in X_train.columns:
                if X_train[col].isna().sum() > 0:
                    if X_train[col].dtype in [np.float64, np.int64]:
                        median_val = X_train[col].median()
                        X_train[col] = X_train[col].fillna(median_val)
                    else:
                        most_common = X_train[col].mode().iloc[0]
                        X_train[col] = X_train[col].fillna(most_common)
            
            # 处理分类变量
            categorical_features = X_train.select_dtypes(include=['object', 'category']).columns
            label_encoders = {}
            
            for cat_feature in categorical_features:
                # 处理分类特征的缺失值
                if X_train[cat_feature].isna().sum() > 0:
                    # 用最常见的值填充缺失值
                    most_common = X_train[cat_feature].mode().iloc[0]
                    X_train[cat_feature] = X_train[cat_feature].fillna(most_common)
                
                le = LabelEncoder()
                X_train[cat_feature] = le.fit_transform(X_train[cat_feature].astype(str))
                label_encoders[cat_feature] = le
            
            # 处理数值型特征中的缺失值（用中位数填充）
            numeric_features = X_train.select_dtypes(include=[np.number]).columns
            for num_feature in numeric_features:
                if X_train[num_feature].isna().sum() > 0:
                    median_val = X_train[num_feature].median()
                    X_train[num_feature] = X_train[num_feature].fillna(median_val)
            
            # 改进的分类变量判断逻辑
            def is_categorical_feature(series, max_categories=20):
                """
                判断特征是否为分类变量
                """
                # 如果是object类型，直接认为是分类变量
                if series.dtype == 'object' or series.dtype == 'category':
                    return True
                
                # 如果是数值型，检查唯一值数量
                unique_count = series.nunique()
                if unique_count <= max_categories:
                    # 进一步检查：如果唯一值数量很少，且都是整数，很可能是分类变量
                    if unique_count <= 10:
                        return True
                    # 检查是否都是整数（允许少量小数，可能是编码后的分类变量）
                    non_null_values = series.dropna()
                    if len(non_null_values) > 0:
                        # 检查是否大部分都是整数
                        integer_count = sum(1 for x in non_null_values if x == int(x))
                        if integer_count / len(non_null_values) > 0.8:
                            return True
                
                return False
            
            # 检查目标变量是否为分类变量
            if is_categorical_feature(y_train):
                # 分类变量，使用逻辑回归
                target_encoder = LabelEncoder()
                y_train_encoded = target_encoder.fit_transform(y_train.astype(str))
                
                # 训练逻辑回归模型
                lr_model = LogisticRegression(random_state=42, max_iter=1000)
                lr_model.fit(X_train, y_train_encoded)
                
                # 预测缺失值
                missing_data = data[missing_mask].copy()
                X_missing = missing_data.drop(columns=[feature])
                
                # 对缺失数据的特征进行相同的预处理
                for cat_feature in categorical_features:
                    if cat_feature in X_missing.columns:
                        # 处理分类特征的缺失值
                        if X_missing[cat_feature].isna().sum() > 0:
                            most_common = X_missing[cat_feature].mode().iloc[0]
                            X_missing[cat_feature] = X_missing[cat_feature].fillna(most_common)
                        X_missing[cat_feature] = label_encoders[cat_feature].transform(
                            X_missing[cat_feature].astype(str)
                        )
                
                for num_feature in numeric_features:
                    if num_feature in X_missing.columns and X_missing[num_feature].isna().sum() > 0:
                        median_val = X_missing[num_feature].median()
                        X_missing[num_feature] = X_missing[num_feature].fillna(median_val)
                
                # 预测
                predictions_encoded = lr_model.predict(X_missing)
                predictions = target_encoder.inverse_transform(predictions_encoded)
                
                # 填充缺失值，确保数据类型兼容
                try:
                    # 尝试转换为与原始列相同的数据类型
                    original_dtype = data[feature].dtype
                    if pd.api.types.is_numeric_dtype(original_dtype):
                        # 对于分类变量，确保预测结果是整数
                        predictions_converted = pd.to_numeric(predictions, errors='coerce').astype(int)
                    else:
                        predictions_converted = predictions.astype(str)
                    
                    imputed_data.loc[missing_mask, feature] = predictions_converted
                except Exception as e:
                    logging.warning(f"数据类型转换失败，使用原始预测值: {e}")
                    imputed_data.loc[missing_mask, feature] = predictions
                
                # 保存模型信息
                model_info['models'][feature] = {
                    'model_type': 'logistic_regression',
                    'model': lr_model,
                    'label_encoders': label_encoders,
                    'target_encoder': target_encoder,
                    'categorical_features': list(categorical_features),
                    'numeric_features': list(numeric_features),
                    'feature_median_values': {
                        feat: X_train[feat].median() for feat in numeric_features
                    }
                }
                
            else:
                # 数值变量，使用线性回归
                from sklearn.linear_model import LinearRegression
                
                lr_model = LinearRegression()
                lr_model.fit(X_train, y_train)
                
                # 预测缺失值
                missing_data = data[missing_mask].copy()
                X_missing = missing_data.drop(columns=[feature])
                
                # 对缺失数据的特征进行相同的预处理
                for cat_feature in categorical_features:
                    if cat_feature in X_missing.columns:
                        X_missing[cat_feature] = label_encoders[cat_feature].transform(
                            X_missing[cat_feature].astype(str)
                        )
                
                for num_feature in numeric_features:
                    if num_feature in X_missing.columns and X_missing[num_feature].isna().sum() > 0:
                        median_val = X_missing[num_feature].median()
                        X_missing[num_feature] = X_missing[num_feature].fillna(median_val)
                
                # 预测
                predictions = lr_model.predict(X_missing)
                
                # 填充缺失值，确保数据类型兼容
                try:
                    # 尝试转换为与原始列相同的数据类型
                    original_dtype = data[feature].dtype
                    if pd.api.types.is_numeric_dtype(original_dtype):
                        predictions_converted = pd.to_numeric(predictions, errors='coerce')
                    else:
                        predictions_converted = predictions.astype(str)
                    
                    imputed_data.loc[missing_mask, feature] = predictions_converted
                except Exception as e:
                    logging.warning(f"数据类型转换失败，使用原始预测值: {e}")
                    imputed_data.loc[missing_mask, feature] = predictions
                
                # 保存模型信息
                model_info['models'][feature] = {
                    'model_type': 'linear_regression',
                    'model': lr_model,
                    'label_encoders': label_encoders,
                    'categorical_features': list(categorical_features),
                    'numeric_features': list(numeric_features),
                    'feature_median_values': {
                        feat: X_train[feat].median() for feat in numeric_features
                    }
                }
            
            # 记录统计信息
            model_info['feature_stats'][feature] = {
                'missing_count': missing_count,
                'missing_percentage': (missing_count / len(data)) * 100,
                'data_type': 'categorical' if is_categorical_feature(y_train) else 'numeric',
                'unique_values_before': y_train.nunique(),
                'unique_values_after': imputed_data[feature].nunique()
            }
            
            model_info['features_imputed'].append(feature)
            
            logging.info(f"特征 '{feature}' 填补完成: {missing_count} 个缺失值")
        
        # 保存填补后的数据
        imputed_data.to_csv(os.path.join(imputed_dir, 'logistic_imputed_train.csv'), index=False)
        
        # 单独保存每个模型（在移除模型对象之前）
        for feature, model_data in model_info['models'].items():
            model_path = os.path.join(models_dir, f'{feature}_model.pkl')
            try:
                # 确保模型对象存在
                if 'model' not in model_data:
                    logging.error(f"模型数据中缺少 'model' 键: {list(model_data.keys())}")
                    continue
                
                # 创建完整的模型数据副本
                model_data_to_save = {
                    'model': model_data['model'],
                    'model_type': model_data.get('model_type', 'unknown'),
                    'categorical_features': model_data.get('categorical_features', []),
                    'numeric_features': model_data.get('numeric_features', []),
                    'feature_median_values': model_data.get('feature_median_values', {})
                }
                
                # 添加编码器（如果存在）
                if 'label_encoders' in model_data:
                    model_data_to_save['label_encoders'] = model_data['label_encoders']
                if 'target_encoder' in model_data:
                    model_data_to_save['target_encoder'] = model_data['target_encoder']
                
                with open(model_path, 'wb') as f:
                    pickle.dump(model_data_to_save, f)
                logging.info(f"模型 {feature} 保存成功")
            except Exception as e:
                logging.error(f"模型 {feature} 保存失败: {e}")
                # 如果保存失败，尝试只保存模型对象
                try:
                    model_only = {'model': model_data['model']}
                    if 'target_encoder' in model_data:
                        model_only['target_encoder'] = model_data['target_encoder']
                    with open(model_path, 'wb') as f:
                        pickle.dump(model_only, f)
                    logging.info(f"模型 {feature} 简化保存成功")
                except Exception as e2:
                    logging.error(f"模型 {feature} 简化保存也失败: {e2}")
        
        # 保存模型信息（不包含模型对象，因为JSON不能序列化模型对象）
        import copy
        model_info_save = copy.deepcopy(model_info)
        for feature in model_info_save['models']:
            # 移除模型对象，只保存其他信息
            model_info_save['models'][feature].pop('model', None)
        
        with open(os.path.join(imputed_dir, 'model_info.json'), 'w') as f:
            json.dump(model_info_save, f, indent=4, default=str)
        
        logging.info(f"逻辑回归填补完成，结果保存在: {imputed_dir}")
        return imputed_data, model_info
    
    def logistic_regression_impute_test(self, test_data: pd.DataFrame, model_info_path: str = None):
        """
        使用训练好的逻辑回归模型对测试集进行缺失值填补
        
        Args:
            test_data: 测试数据
            model_info_path: 模型信息文件路径，如果为None则使用默认路径
        """
        logging.info("==========logistic_regression_impute_test==========")
        
        if model_info_path is None:
            model_info_path = os.path.join(self.output_dir, 'logistic_imputed', 'model_info.json')
        
        if not os.path.exists(model_info_path):
            raise FileNotFoundError(f"模型信息文件不存在: {model_info_path}")
        
        # 加载模型信息
        with open(model_info_path, 'r') as f:
            model_info = json.load(f)
        
        models_dir = os.path.join(self.output_dir, 'logistic_imputed', 'models')
        imputed_test_data = test_data.copy()
        
        for feature in model_info['features_imputed']:
            logging.info(f"正在填补测试集特征: {feature}")
            
            # 加载模型
            model_path = os.path.join(models_dir, f'{feature}_model.pkl')
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # 检查测试集中是否有该特征的缺失值
            missing_mask = test_data[feature].isna()
            missing_count = missing_mask.sum()
            
            if missing_count == 0:
                logging.info(f"测试集中特征 '{feature}' 没有缺失值，跳过")
                continue
            
            # 准备测试数据
            X_test = test_data.drop(columns=[feature])
            
            # 对测试数据进行相同的预处理
            categorical_features = model_data['categorical_features']
            numeric_features = model_data['numeric_features']
            label_encoders = model_data['label_encoders']
            
            for cat_feature in categorical_features:
                if cat_feature in X_test.columns:
                    # 处理训练时未见过的类别
                    unique_train_values = set(label_encoders[cat_feature].classes_)
                    unique_test_values = set(X_test[cat_feature].unique())
                    new_values = unique_test_values - unique_train_values
                    
                    if new_values:
                        logging.warning(f"特征 '{cat_feature}' 在测试集中有新的类别: {new_values}")
                        # 将新类别映射为最常见的类别
                        most_common = X_test[cat_feature].mode().iloc[0]
                        X_test[cat_feature] = X_test[cat_feature].replace(list(new_values), most_common)
                    
                    X_test[cat_feature] = label_encoders[cat_feature].transform(X_test[cat_feature].astype(str))
            
            for num_feature in numeric_features:
                if num_feature in X_test.columns and X_test[num_feature].isna().sum() > 0:
                    median_val = model_data['feature_median_values'][num_feature]
                    X_test[num_feature] = X_test[num_feature].fillna(median_val)
            
            # 预测缺失值
            if 'model' not in model_data:
                logging.error(f"模型数据中缺少 'model' 键: {list(model_data.keys())}")
                continue
                
            if model_data.get('model_type') == 'logistic_regression':
                predictions_encoded = model_data['model'].predict(X_test[missing_mask])
                if 'target_encoder' in model_data:
                    predictions = model_data['target_encoder'].inverse_transform(predictions_encoded)
                else:
                    predictions = predictions_encoded
                
                # 对于逻辑回归预测的分类变量，确保结果是整数
                try:
                    predictions = pd.to_numeric(predictions, errors='coerce').astype(int)
                except Exception as e:
                    logging.warning(f"分类变量预测结果转换为整数失败: {e}")
            else:  # linear_regression
                predictions = model_data['model'].predict(X_test[missing_mask])
            
            # 填充缺失值
            imputed_test_data.loc[missing_mask, feature] = predictions
            
            logging.info(f"测试集特征 '{feature}' 填补完成: {missing_count} 个缺失值")
        
        # 保存填补后的测试数据
        test_output_dir = os.path.join(self.output_dir, 'logistic_imputed')
        imputed_test_data.to_csv(os.path.join(test_output_dir, 'logistic_imputed_test.csv'), index=False)
        
        logging.info(f"测试集逻辑回归填补完成")
        return imputed_test_data
    
    def analyse_knn_impute(self):
        """
        分析KNN填补缺失值后的数据，统计每个特征的缺失值数量，填值前后的最大值，最小值，平均值，中位数，标准差，方差，偏度，峰度
        """
        logging.info("==========analyse_knn_impute==========")
        
        # 创建输出目录
        imputed_dir = os.path.join(self.output_dir, 'imputed')
        os.makedirs(imputed_dir, exist_ok=True)
        
        # 获取原始数据（填补前）
        original_data = pd.read_csv(self.output_dir/'recoded_train.csv',index_col=0)
        imputed_data = pd.read_csv(self.output_dir/'knn_imputed_train.csv',index_col=0)
        # 前后数据集的列明区别
        original_columns = original_data.columns
        imputed_columns = imputed_data.columns

        for col in original_columns:
            if col not in imputed_columns:
                logging.warning(f"特征 '{col}' 在填补后的数据中不存在，跳过")
                continue
        
        for col in imputed_columns:
            if col not in original_columns:
                logging.warning(f"特征 '{col}' 在原始数据中不存在，跳过")
                continue

        # 初始化结果DataFrame
        res = pd.DataFrame(columns=['feature','missing_count','max_value(before)','max_value(after)','min_value(before)','min_value(after)','mean_value(before)','mean_value(after)','median_value(before)','median_value(after)','std_value(before)','std_value(after)','var_value(before)','var_value(after)','skewness(before)','skewness(after)','kurtosis(before)','kurtosis(after)'])
        
        for feature in imputed_data.columns:
            if feature not in original_data.columns:
                logging.warning(f"特征 '{feature}' 在原始数据中不存在，跳过")
                continue
                
            # 计算原始数据中的缺失值数量
            missing_count = original_data[feature].isna().sum()
            
            # 如果该特征没有缺失值，跳过
            if missing_count == 0:
                continue
            
            # 获取填补前的数据（非缺失值）
            before_data = original_data[feature].dropna()
            
            # 获取填补后的数据
            after_data = imputed_data[feature]
            
            if len(before_data) == 0:
                logging.warning(f"特征 '{feature}' 填补前没有有效数据")
                continue
            
            # 计算填补前的统计量
            max_value_before = float(before_data.max())
            min_value_before = float(before_data.min())
            mean_value_before = float(before_data.mean())
            median_value_before = float(before_data.median())
            std_value_before = float(before_data.std())
            var_value_before = float(before_data.var())
            skewness_before = float(before_data.skew())
            kurtosis_before = float(before_data.kurtosis())
            
            # 计算填补后的统计量
            max_value_after = float(after_data.max())
            min_value_after = float(after_data.min())
            mean_value_after = float(after_data.mean())
            median_value_after = float(after_data.median())
            std_value_after = float(after_data.std())
            var_value_after = float(after_data.var())
            skewness_after = float(after_data.skew())
            kurtosis_after = float(after_data.kurtosis())
            
            # 添加到结果DataFrame
            new_row = pd.DataFrame({
                'feature': [feature],
                'missing_count': [missing_count],
                'max_value(before)': [max_value_before],
                'max_value(after)': [max_value_after],
                'min_value(before)': [min_value_before],
                'min_value(after)': [min_value_after],
                'mean_value(before)': [mean_value_before],
                'mean_value(after)': [mean_value_after],
                'median_value(before)': [median_value_before],
                'median_value(after)': [median_value_after],
                'std_value(before)': [std_value_before],
                'std_value(after)': [std_value_after],
                'var_value(before)': [var_value_before],
                'var_value(after)': [var_value_after],
                'skewness(before)': [skewness_before],
                'skewness(after)': [skewness_after],
                'kurtosis(before)': [kurtosis_before],
                'kurtosis(after)': [kurtosis_after]
            })
            
            res = pd.concat([res, new_row], ignore_index=True)
            
            logging.info(f"特征 '{feature}': {missing_count} 个缺失值")
        
        # 按缺失值数量排序
        res = res.sort_values('missing_count', ascending=False)
        
        # 保存结果
        output_path = os.path.join(imputed_dir, 'knn_imputation_analysis.csv')
        res.to_csv(output_path, index=False)
        
        # 保存详细统计信息
        detailed_stats = {
            'total_features_analyzed': len(res),
            'total_missing_values_filled': res['missing_count'].sum(),
            'features_with_missing_values': res['feature'].tolist(),
            'missing_count_summary': res['missing_count'].describe().to_dict(),
            'statistics_comparison': {
                'max_value_change': (res['max_value(after)'] - res['max_value(before)']).describe().to_dict(),
                'min_value_change': (res['min_value(after)'] - res['min_value(before)']).describe().to_dict(),
                'mean_value_change': (res['mean_value(after)'] - res['mean_value(before)']).describe().to_dict(),
                'std_value_change': (res['std_value(after)'] - res['std_value(before)']).describe().to_dict(),
                'skewness_change': (res['skewness(after)'] - res['skewness(before)']).describe().to_dict(),
                'kurtosis_change': (res['kurtosis(after)'] - res['kurtosis(before)']).describe().to_dict()
            }
        }
        
        detailed_output_path = os.path.join(imputed_dir, 'knn_imputation_detailed_stats.json')
        with open(detailed_output_path, 'w') as f:
            json.dump(detailed_stats, f, indent=4)
        
        logging.info(f"分析完成，结果保存在: {imputed_dir}")
        logging.info(f"分析了 {len(res)} 个有缺失值的特征")
        logging.info(f"总共填补了 {res['missing_count'].sum()} 个缺失值")
        
        return res

    def analyze_logistic_imputation(self, original_data: pd.DataFrame = None):
        """
        分析逻辑回归填补缺失值后的数据
        """
        logging.info("==========analyze_logistic_imputation==========")
        
        # 创建输出目录
        imputed_dir = os.path.join(self.output_dir, 'logistic_imputed')
        os.makedirs(imputed_dir, exist_ok=True)
        
        # 获取原始数据（填补前）
        if original_data is None:
            original_data = pd.read_csv(self.output_dir/'recoded_train.csv', index_col=0)
        
        imputed_data = pd.read_csv(os.path.join(imputed_dir, 'logistic_imputed_train.csv'), index_col=0)
        
        # 初始化结果DataFrame
        res = pd.DataFrame(columns=['feature','missing_count','max_value(before)','max_value(after)','min_value(before)','min_value(after)','mean_value(before)','mean_value(after)','median_value(before)','median_value(after)','std_value(before)','std_value(after)','var_value(before)','var_value(after)','skewness(before)','skewness(after)','kurtosis(before)','kurtosis(after)'])
        
        for feature in imputed_data.columns:
            if feature not in original_data.columns:
                logging.warning(f"特征 '{feature}' 在原始数据中不存在，跳过")
                continue
                
            # 计算原始数据中的缺失值数量
            missing_count = original_data[feature].isna().sum()
            
            # 如果该特征没有缺失值，跳过
            if missing_count == 0:
                continue
            
            # 获取填补前的数据（非缺失值）
            before_data = original_data[feature].dropna()
            
            # 获取填补后的数据
            after_data = imputed_data[feature]
            
            if len(before_data) == 0:
                logging.warning(f"特征 '{feature}' 填补前没有有效数据")
                continue
            
            # 计算填补前的统计量
            max_value_before = float(before_data.max())
            min_value_before = float(before_data.min())
            mean_value_before = float(before_data.mean())
            median_value_before = float(before_data.median())
            std_value_before = float(before_data.std())
            var_value_before = float(before_data.var())
            skewness_before = float(before_data.skew())
            kurtosis_before = float(before_data.kurtosis())
            
            # 计算填补后的统计量
            max_value_after = float(after_data.max())
            min_value_after = float(after_data.min())
            mean_value_after = float(after_data.mean())
            median_value_after = float(after_data.median())
            std_value_after = float(after_data.std())
            var_value_after = float(after_data.var())
            skewness_after = float(after_data.skew())
            kurtosis_after = float(after_data.kurtosis())
            
            # 添加到结果DataFrame
            new_row = pd.DataFrame({
                'feature': [feature],
                'missing_count': [missing_count],
                'max_value(before)': [max_value_before],
                'max_value(after)': [max_value_after],
                'min_value(before)': [min_value_before],
                'min_value(after)': [min_value_after],
                'mean_value(before)': [mean_value_before],
                'mean_value(after)': [mean_value_after],
                'median_value(before)': [median_value_before],
                'median_value(after)': [median_value_after],
                'std_value(before)': [std_value_before],
                'std_value(after)': [std_value_after],
                'var_value(before)': [var_value_before],
                'var_value(after)': [var_value_after],
                'skewness(before)': [skewness_before],
                'skewness(after)': [skewness_after],
                'kurtosis(before)': [kurtosis_before],
                'kurtosis(after)': [kurtosis_after]
            })
            
            res = pd.concat([res, new_row], ignore_index=True)
            
            logging.info(f"特征 '{feature}': {missing_count} 个缺失值")
        
        # 按缺失值数量排序
        res = res.sort_values('missing_count', ascending=False)
        
        # 保存结果
        output_path = os.path.join(imputed_dir, 'logistic_imputation_analysis.csv')
        res.to_csv(output_path, index=False)
        
        # 保存详细统计信息
        detailed_stats = {
            'total_features_analyzed': len(res),
            'total_missing_values_filled': res['missing_count'].sum(),
            'features_with_missing_values': res['feature'].tolist(),
            'missing_count_summary': res['missing_count'].describe().to_dict(),
            'statistics_comparison': {
                'max_value_change': (res['max_value(after)'] - res['max_value(before)']).describe().to_dict(),
                'min_value_change': (res['min_value(after)'] - res['min_value(before)']).describe().to_dict(),
                'mean_value_change': (res['mean_value(after)'] - res['mean_value(before)']).describe().to_dict(),
                'std_value_change': (res['std_value(after)'] - res['std_value(before)']).describe().to_dict(),
                'skewness_change': (res['skewness(after)'] - res['skewness(before)']).describe().to_dict(),
                'kurtosis_change': (res['kurtosis(after)'] - res['kurtosis(before)']).describe().to_dict()
            }
        }
        
        detailed_output_path = os.path.join(imputed_dir, 'logistic_imputation_detailed_stats.json')
        with open(detailed_output_path, 'w') as f:
            json.dump(detailed_stats, f, indent=4)
        
        logging.info(f"逻辑回归填补分析完成，结果保存在: {imputed_dir}")
        logging.info(f"分析了 {len(res)} 个有缺失值的特征")
        logging.info(f"总共填补了 {res['missing_count'].sum()} 个缺失值")
        
        return res

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
    handler.load_data(mode='recoded')
    handler.analyze_missing_values()
    # data = handler.train_data
    # # handler.pca_for_low_mid_missing_dataset() # 主成分分析
    handler.load_data(mode='normalized')
    # data = handler.drop_mid_missing_dataset(data)
    # data = handler.drop_low_missing_dataset(data)
    data = handler.knn_impute(data)
    data.to_csv(os.path.join(handler.output_dir, 'knn_imputed_train.csv'), index=False)
    
    # 分析KNN填补结果
    analysis_result = handler.analyse_knn_impute()
    print(f"KNN填补分析完成，分析了 {len(analysis_result)} 个特征")


if __name__ == '__main__':
    main()
