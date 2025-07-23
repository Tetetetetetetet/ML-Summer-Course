#!/usr/bin/env python3
"""
改进的逻辑回归缺失值填补方法
解决迭代限制问题
"""

import pandas as pd
import numpy as np
import logging
import os
import json
import pickle
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)

class ImprovedLogisticImputation:
    def __init__(self, output_dir='Dataset/processed/train_processed'):
        self.output_dir = Path(output_dir)
        self.scalers = {}
        self.models = {}
        self.label_encoders = {}
        self.last_train_data = None
        self.original_train_shape = None
        self.original_test_shape = None
        self.original_train_missing = 0
        self.original_test_missing = 0
        
    def is_categorical_feature(self, series, max_categories=20):
        """判断是否为分类特征"""
        if series.dtype in ['object', 'category']:
            return True
        elif series.dtype in [np.int64, np.int32]:
            # 检查唯一值数量
            unique_count = series.nunique()
            return unique_count <= max_categories
        return False
    
    def preprocess_features(self, X_train, X_missing=None, feature_name=None):
        """
        改进的特征预处理方法
        """
        processed_X_train = X_train.copy()
        processed_X_missing = X_missing.copy() if X_missing is not None else None
        
        # 1. 处理分类变量
        categorical_features = X_train.select_dtypes(include=['object', 'category']).columns
        for cat_feature in categorical_features:
            # 处理训练数据
            if X_train[cat_feature].isna().sum() > 0:
                most_common = X_train[cat_feature].mode().iloc[0]
                processed_X_train[cat_feature] = processed_X_train[cat_feature].fillna(most_common)
            
            # 标签编码
            le = LabelEncoder()
            processed_X_train[cat_feature] = le.fit_transform(processed_X_train[cat_feature].astype(str))
            self.label_encoders[cat_feature] = le
            
            # 处理缺失数据（如果存在）
            if processed_X_missing is not None and cat_feature in processed_X_missing.columns:
                if processed_X_missing[cat_feature].isna().sum() > 0:
                    most_common = processed_X_missing[cat_feature].mode().iloc[0]
                    processed_X_missing[cat_feature] = processed_X_missing[cat_feature].fillna(most_common)
                
                # 处理未见过的类别
                unique_train_values = set(le.classes_)
                unique_missing_values = set(processed_X_missing[cat_feature].unique())
                new_values = unique_missing_values - unique_train_values
                
                if new_values:
                    logging.warning(f"特征 '{cat_feature}' 有新的类别: {new_values}")
                    most_common = processed_X_missing[cat_feature].mode().iloc[0]
                    processed_X_missing[cat_feature] = processed_X_missing[cat_feature].replace(list(new_values), most_common)
                
                processed_X_missing[cat_feature] = le.transform(processed_X_missing[cat_feature].astype(str))
        
        # 2. 处理数值变量
        numeric_features = X_train.select_dtypes(include=[np.number]).columns
        for num_feature in numeric_features:
            # 处理训练数据
            if X_train[num_feature].isna().sum() > 0:
                median_val = X_train[num_feature].median()
                processed_X_train[num_feature] = processed_X_train[num_feature].fillna(median_val)
            
            # 处理缺失数据（如果存在）
            if processed_X_missing is not None and num_feature in processed_X_missing.columns:
                if processed_X_missing[num_feature].isna().sum() > 0:
                    median_val = processed_X_missing[num_feature].median()
                    processed_X_missing[num_feature] = processed_X_missing[num_feature].fillna(median_val)
        
        # 3. 标准化数值特征
        if len(numeric_features) > 0:
            scaler = StandardScaler()
            processed_X_train[numeric_features] = scaler.fit_transform(processed_X_train[numeric_features])
            self.scalers[feature_name] = scaler
            
            if processed_X_missing is not None:
                processed_X_missing[numeric_features] = scaler.transform(processed_X_missing[numeric_features])
        
        return processed_X_train, processed_X_missing
    
    def train_robust_model(self, X_train, y_train, is_categorical=True):
        """
        训练鲁棒的模型，处理收敛问题
        """
        if is_categorical:
            # 尝试不同的求解器和参数
            solvers = ['lbfgs', 'liblinear', 'saga']
            max_iters = [2000, 5000, 10000]
            
            for solver in solvers:
                for max_iter in max_iters:
                    try:
                        if solver == 'liblinear':
                            model = LogisticRegression(
                                solver=solver, 
                                max_iter=max_iter, 
                                random_state=42,
                                C=1.0,
                                tol=1e-4
                            )
                        else:
                            model = LogisticRegression(
                                solver=solver, 
                                max_iter=max_iter, 
                                random_state=42,
                                C=1.0,
                                tol=1e-4
                            )
                        
                        model.fit(X_train, y_train)
                        logging.info(f"成功训练模型: solver={solver}, max_iter={max_iter}")
                        return model
                        
                    except Exception as e:
                        logging.warning(f"模型训练失败: solver={solver}, max_iter={max_iter}, error={e}")
                        continue
            
            # 如果所有尝试都失败，使用最简单的模型
            logging.warning("所有逻辑回归模型都失败，使用线性回归作为备选")
            return LinearRegression()
        else:
            # 数值变量使用线性回归
            return LinearRegression()
    
    def impute_feature(self, data, feature, missing_mask):
        """
        对单个特征进行填补
        """
        logging.info(f"正在填补特征: {feature}")
        
        # 获取训练数据
        train_mask = ~missing_mask
        train_data = data[train_mask].copy()
        
        # 保存训练数据用于测试集处理
        if self.last_train_data is None:
            self.last_train_data = data.copy()
        
        if train_data.shape[0] == 0:
            logging.warning(f"特征 {feature} 没有可用样本，跳过填补")
            return data
        
        # 准备特征和目标变量
        X_train = train_data.drop(columns=[feature])
        y_train = train_data[feature]
        
        # 判断目标变量类型
        is_categorical = self.is_categorical_feature(y_train)
        if not is_categorical:
            logging.info(f'特征 {feature} 的目标变量类型: 是数值类型')
            input("enter to continue")
        
        # 预处理特征
        X_train_processed, _ = self.preprocess_features(X_train, feature_name=feature)
        
        # 处理目标变量
        if is_categorical:
            target_encoder = LabelEncoder()
            y_train_encoded = target_encoder.fit_transform(y_train.astype(str))
            self.label_encoders[feature] = target_encoder
        else:
            y_train_encoded = y_train
        
        # 训练模型
        model = self.train_robust_model(X_train_processed, y_train_encoded, is_categorical)
        self.models[feature] = model
        
        # 预测缺失值
        missing_data = data[missing_mask].copy()
        X_missing = missing_data.drop(columns=[feature])
        
        # 预处理缺失数据
        X_train_for_preprocessing = data[train_mask].drop(columns=[feature])
        _, X_missing_processed = self.preprocess_features(X_train_for_preprocessing, X_missing, feature_name=feature)
        
        # 预测
        if is_categorical and hasattr(model, 'predict'):
            try:
                predictions_encoded = model.predict(X_missing_processed)
                predictions = target_encoder.inverse_transform(predictions_encoded)
            except Exception as e:
                logging.warning(f"模型预测失败，使用众数填充: {e}")
                most_common = y_train.mode().iloc[0]
                predictions = [most_common] * missing_mask.sum()
        else:
            try:
                predictions = model.predict(X_missing_processed)
            except Exception as e:
                logging.warning(f"模型预测失败，使用中位数填充: {e}")
                median_val = y_train.median()
                predictions = [median_val] * missing_mask.sum()
        
        # 填充缺失值
        imputed_data = data.copy()
        try:
            original_dtype = data[feature].dtype
            if pd.api.types.is_numeric_dtype(original_dtype):
                if is_categorical:
                    predictions_converted = pd.to_numeric(predictions, errors='coerce').astype(int)
                else:
                    predictions_converted = pd.to_numeric(predictions, errors='coerce')
            else:
                predictions_converted = predictions.astype(str)
            
            imputed_data.loc[missing_mask, feature] = predictions_converted
        except Exception as e:
            logging.warning(f"数据类型转换失败，使用原始预测值: {e}")
            imputed_data.loc[missing_mask, feature] = predictions
        
        logging.info(f"特征 '{feature}' 填补完成: {missing_mask.sum()} 个缺失值")
        return imputed_data
    
    def impute_dataset(self, train_data, test_data=None):
        """
        对整个数据集进行填补
        
        Args:
            train_data: 训练数据
            test_data: 测试数据（可选）
        """
        logging.info("==========开始改进的逻辑回归填补==========")
        
        # 记录原始数据信息
        self.original_train_shape = train_data.shape
        self.original_train_missing = train_data.isnull().sum().sum()
        if test_data is not None:
            self.original_test_shape = test_data.shape
            self.original_test_missing = test_data.isnull().sum().sum()
        
        # 创建输出目录
        imputed_dir = self.output_dir / 'improved_logistic_imputed'
        imputed_dir.mkdir(exist_ok=True)
        
        # 获取有缺失值的特征
        missing_features = []
        for col in train_data.columns:
            if train_data[col].isna().sum() > 0:
                missing_features.append(col)
        
        logging.info(f"需要填补的特征: {missing_features}")
        
        imputed_train = train_data.copy()
        
        # 逐个填补训练集特征
        for feature in missing_features:
            missing_mask = train_data[feature].isna()
            if missing_mask.sum() > 0:
                imputed_train = self.impute_feature(imputed_train, feature, missing_mask)
        
        # 处理测试集
        imputed_test = None
        if test_data is not None:
            logging.info("开始填补测试集...")
            imputed_test = test_data.copy()
            
            # 使用训练好的模型填补测试集
            for feature in missing_features:
                if feature in self.models:
                    missing_mask = test_data[feature].isna()
                    if missing_mask.sum() > 0:
                        imputed_test = self.impute_test_feature(imputed_test, feature, missing_mask)
        
        # 保存结果
        imputed_train.to_csv(imputed_dir / 'improved_logistic_imputed_train_final.csv', index=False)
        if imputed_test is not None:
            imputed_test.to_csv(imputed_dir / 'improved_logistic_imputed_test_final.csv', index=False)
        
        # 保存模型信息
        model_info = {
            'imputation_method': 'improved_logistic_regression',
            'features_imputed': missing_features,
            'model_count': len(self.models),
            'scaler_count': len(self.scalers),
            'encoder_count': len(self.label_encoders)
        }
        
        with open(imputed_dir / 'model_info.json', 'w') as f:
            json.dump(model_info, f, indent=4, default=str)
        
        # 保存模型对象
        models_dir = imputed_dir / 'models'
        models_dir.mkdir(exist_ok=True)
        
        for feature, model in self.models.items():
            model_data = {
                'model': model,
                'scaler': self.scalers.get(feature),
                'label_encoders': {k: v for k, v in self.label_encoders.items() if k != feature}
            }
            
            with open(models_dir / f'{feature}_model.pkl', 'wb') as f:
                pickle.dump(model_data, f)
        
        # 生成报告
        self.generate_imputation_report(imputed_train, imputed_test, imputed_dir)
        
        logging.info(f"改进的逻辑回归填补完成，结果保存在: {imputed_dir}")
        return imputed_train, imputed_test
    
    def impute_test_feature(self, test_data, feature, missing_mask):
        """
        使用训练好的模型填补测试集特征
        """
        logging.info(f"正在填补测试集特征: {feature}")
        
        if feature not in self.models:
            logging.warning(f"特征 '{feature}' 没有训练好的模型，跳过")
            return test_data
        
        # 准备测试数据
        X_test = test_data.drop(columns=[feature])
        
        # 预处理测试数据
        X_train_for_preprocessing = self.last_train_data.drop(columns=[feature])
        _, X_test_processed = self.preprocess_features(X_train_for_preprocessing, X_test, feature_name=feature)
        
        # 预测
        model = self.models[feature]
        if hasattr(model, 'predict'):
            try:
                predictions = model.predict(X_test_processed)
                
                # 如果是分类变量，需要反向转换
                if feature in self.label_encoders:
                    predictions = self.label_encoders[feature].inverse_transform(predictions)
                
                # 填充缺失值
                test_data.loc[missing_mask, feature] = predictions
                
            except Exception as e:
                logging.warning(f"模型预测失败，使用众数填充: {e}")
                most_common = self.last_train_data[feature].mode().iloc[0]
                test_data.loc[missing_mask, feature] = most_common
        
        logging.info(f"测试集特征 '{feature}' 填补完成: {missing_mask.sum()} 个缺失值")
        return test_data
    
    def generate_imputation_report(self, imputed_train, imputed_test, imputed_dir):
        """
        生成填值报告
        """
        report = {
            'imputation_method': 'improved_logistic_regression',
            'train_data_info': {
                'original_shape': list(self.original_train_shape),
                'imputed_shape': list(imputed_train.shape),
                'original_missing': str(self.original_train_missing),
                'imputed_missing': str(imputed_train.isnull().sum().sum())
            },
            'test_data_info': {
                'original_shape': list(self.original_test_shape) if imputed_test is not None else None,
                'imputed_shape': list(imputed_test.shape) if imputed_test is not None else None,
                'original_missing': str(self.original_test_missing) if imputed_test is not None else None,
                'imputed_missing': str(imputed_test.isnull().sum().sum()) if imputed_test is not None else None
            },
            'features_processed': len(self.models),
            'processing_timestamp': datetime.now().isoformat()
        }
        
        with open(imputed_dir / 'imputation_report.json', 'w') as f:
            json.dump(report, f, indent=4, default=str)

def main():
    """主函数"""
    logging.info("==========改进的逻辑回归填补流程开始==========")
    
    # 初始化处理器
    handler = ImprovedLogisticImputation()
    
    # 加载训练数据
    train_path = handler.output_dir / 'recoded_train.csv'
    if not train_path.exists():
        logging.error(f"训练数据文件不存在: {train_path}")
        return
    
    train_data = pd.read_csv(train_path)
    logging.info(f"加载训练数据: {train_path}, 形状: {train_data.shape}")
    
    # 加载测试数据
    test_path = Path('Dataset/processed/test_processed/recoded_test.csv')
    test_data = None
    if test_path.exists():
        test_data = pd.read_csv(test_path)
        logging.info(f"加载测试数据: {test_path}, 形状: {test_data.shape}")
    else:
        logging.warning(f"测试数据文件不存在: {test_path}")
    
    # 分析缺失值
    train_missing = train_data.isnull().sum().sum()
    test_missing = test_data.isnull().sum().sum() if test_data is not None else 0
    logging.info(f"训练集缺失值: {train_missing}")
    logging.info(f"测试集缺失值: {test_missing}")
    
    # 进行填补
    imputed_train, imputed_test = handler.impute_dataset(train_data, test_data)
    
    # 验证结果
    final_train_missing = imputed_train.isnull().sum().sum()
    final_test_missing = imputed_test.isnull().sum().sum() if imputed_test is not None else 0
    logging.info(f"填补后训练集缺失值: {final_train_missing}")
    logging.info(f"填补后测试集缺失值: {final_test_missing}")
    
    logging.info("==========改进的逻辑回归填补流程完成==========")

if __name__ == "__main__":
    main() 