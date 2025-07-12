import pandas as pd
import numpy as np
import logging
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import LabelEncoder
import os

logging.basicConfig(level=logging.INFO)

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

def test_logistic_imputation_simple(data, target_feature=None):
    """
    简化的逻辑回归填充函数，不依赖外部配置
    """
    logging.info("==========简化的逻辑回归填充测试==========")
    
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
        
        # 准备特征和目标变量
        X_train = train_data.drop(columns=[feature])
        y_train = train_data[feature]
        
        # 处理分类变量
        categorical_features = X_train.select_dtypes(include=['object', 'category']).columns
        label_encoders = {}
        
        for cat_feature in categorical_features:
            le = LabelEncoder()
            X_train[cat_feature] = le.fit_transform(X_train[cat_feature].astype(str))
            label_encoders[cat_feature] = le
        
        # 处理数值型特征中的缺失值（用中位数填充）
        numeric_features = X_train.select_dtypes(include=[np.number]).columns
        for num_feature in numeric_features:
            if X_train[num_feature].isna().sum() > 0:
                median_val = X_train[num_feature].median()
                X_train[num_feature] = X_train[num_feature].fillna(median_val)
        
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
            
        else:
            # 数值变量，使用线性回归
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
        
        logging.info(f"特征 '{feature}' 填补完成: {missing_count} 个缺失值")
    
    return imputed_data

def test_categorical_imputation():
    """
    测试分类变量填充是否正确
    """
    # 创建测试数据
    np.random.seed(42)
    n_samples = 100
    
    # 创建包含分类变量的测试数据
    data = pd.DataFrame({
        'category_feature': np.random.choice(['A', 'B', 'C'], size=n_samples),
        'numeric_feature': np.random.normal(0, 1, n_samples),
        'binary_feature': np.random.choice([0, 1], size=n_samples),
        'multi_category': np.random.choice([1, 2, 3, 4, 5], size=n_samples)
    })
    
    # 添加一些缺失值
    missing_indices = np.random.choice(n_samples, size=20, replace=False)
    data.loc[missing_indices, 'category_feature'] = np.nan
    
    missing_indices = np.random.choice(n_samples, size=15, replace=False)
    data.loc[missing_indices, 'binary_feature'] = np.nan
    
    missing_indices = np.random.choice(n_samples, size=10, replace=False)
    data.loc[missing_indices, 'multi_category'] = np.nan
    
    print("原始数据:")
    print(data.head())
    print("\n缺失值统计:")
    print(data.isna().sum())
    
    print("\n分类变量判断测试:")
    for col in data.columns:
        is_cat = is_categorical_feature(data[col])
        print(f"{col}: {'分类变量' if is_cat else '数值变量'} (dtype: {data[col].dtype}, unique: {data[col].nunique()})")
    
    # 测试填充
    print("\n开始填充测试...")
    imputed_data = test_logistic_imputation_simple(data)
    
    print("\n填充后数据:")
    print(imputed_data.head())
    print("\n填充后缺失值统计:")
    print(imputed_data.isna().sum())
    
    print("\n填充后数据类型:")
    for col in imputed_data.columns:
        print(f"{col}: {imputed_data[col].dtype}")
        if is_categorical_feature(data[col]):
            print(f"  - 原始唯一值: {data[col].dropna().unique()}")
            print(f"  - 填充后唯一值: {imputed_data[col].unique()}")
    
    # 检查分类变量是否还是整数
    print("\n分类变量检查:")
    for col in ['category_feature', 'binary_feature', 'multi_category']:
        if is_categorical_feature(data[col]):
            # 检查是否有小数
            has_decimals = any(not float(x).is_integer() for x in imputed_data[col] if pd.notna(x))
            print(f"{col}: {'包含小数' if has_decimals else '都是整数'}")
            
            # 检查是否都是原始类别
            original_values = set(data[col].dropna().unique())
            imputed_values = set(imputed_data[col].unique())
            print(f"  - 原始类别: {original_values}")
            print(f"  - 填充后类别: {imputed_values}")
            print(f"  - 是否包含新类别: {bool(imputed_values - original_values)}")
    
    # 额外检查：查看具体的填充值
    print("\n具体填充值检查:")
    for col in ['category_feature', 'binary_feature', 'multi_category']:
        if is_categorical_feature(data[col]):
            print(f"\n{col} 的填充值:")
            missing_mask = data[col].isna()
            filled_values = imputed_data.loc[missing_mask, col]
            print(f"  填充了 {len(filled_values)} 个值")
            print(f"  填充值: {filled_values.values}")
            print(f"  填充值类型: {filled_values.dtype}")

if __name__ == "__main__":
    test_categorical_imputation() 