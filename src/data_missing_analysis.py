import pandas as pd
import numpy as np
import pdb
from sklearn.covariance import MinCovDet
from scipy.stats import multivariate_normal
from sklearn.preprocessing import StandardScaler
import json
from myutils import read_jsonl

class MissingAnalysisHandler:
    def __init__(self):
        self.feature_json = read_jsonl('config/feature.json')    
        self.analysis_data = pd.DataFrame()
        self.is_categorical = []
        self.missing_features = []
    
    def get_feature(self, data : pd.DataFrame):
        for col in data.columns:
            if data[col].isna().sum() > 0:
                self.missing_features.append(col)
            else :
                self.analysis_data[col] = data[col]
            if self.feature_json['feature'][col]['type'] == 'categorical':
                self.is_categorical[col] = True
            else :
                self.is_categorical = False
    
    def fast_robust_cov(X):
        """快速鲁棒协方差估计"""
        # 计算鲁棒中心（中位数）
        center = np.median(X, axis=0)
        
        # 计算绝对偏差
        devs = np.abs(X - center)
        
        # 计算鲁棒尺度（MAD）
        scales = np.median(devs, axis=0) / 0.6745  # 转换为标准差估计
        
        # 构建对角协方差矩阵
        cov = np.diag(scales**2)
        
        return cov

    def em_imputation_df(self, data : pd.DataFrame, max_iter=100, tol=1e-5, robust_cov=True):
        robust_cov_type="subsample"
        subsample_size=5000
        print("now1")
        df = data.copy(deep=True)
        original_columns = df.columns.tolist()
        original_index = df.index
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        n_features = len(numeric_cols)
        non_numeric_cols = [col for col in df.columns if col not in numeric_cols]
        print("now2")
        scaler = None
        if len(numeric_cols) > 0:
            scaler = StandardScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        
        X = df.values
        nan_mask = np.isnan(X)
        col_means = np.nanmean(X, axis=0)
        print("now3")
        for col_idx in range(X.shape[1]):
            col_mask = nan_mask[:, col_idx]
            if col_idx < len(numeric_cols):
                X[col_mask, col_idx] = col_means[col_idx]
            else:  
                print("出现了奇异类型")
        
        mu = np.nanmean(X, axis=0)
        print("now4")
        if robust_cov and len(numeric_cols) > 0:
            numeric_idx = [df.columns.get_loc(c) for c in numeric_cols]
            X_numeric = X[:, numeric_idx]
            
            if robust_cov_type == "full":
                try:
                    robust_cov_mat = MinCovDet(
                        support_fraction=0.8,
                        max_iter=20,
                        random_state=42
                    ).fit(X_numeric).covariance_
                except:
                    robust_cov_mat = np.cov(X_numeric, rowvar=False)
            
            elif robust_cov_type == "subsample":
                if X_numeric.shape[0] > subsample_size:
                    subsample = X_numeric[np.random.choice(X_numeric.shape[0], subsample_size, replace=False)]
                else:
                    subsample = X_numeric
                    
                robust_cov_mat = MinCovDet(
                    support_fraction=0.8
                ).fit(subsample).covariance_
            
            elif robust_cov_type == "fast":
                robust_cov_mat = self.fast_robust_cov(X_numeric)
            
            cov = np.eye(n_features)
            for i, idx_i in enumerate(numeric_idx):
                for j, idx_j in enumerate(numeric_idx):
                    cov[idx_i, idx_j] = robust_cov_mat[i, j]
        
        else:
            cov = np.cov(X, rowvar=False, bias=True) + np.eye(n_features) * 1e-6


        print("now5")
        cov += np.eye(cov.shape[0]) * 1e-6
        
        prev_log_likelihood = -np.inf
        log_likelihoods = []
    
        for iteration in range(max_iter):
            print(iteration)
            for i in range(len(X)):
                mis_idx = nan_mask[i, :]
                obs_idx = ~mis_idx
                
                if np.any(mis_idx):
                    mu_obs = mu[obs_idx]
                    mu_mis = mu[mis_idx]
                    
                    cov_obs_obs = cov[np.ix_(obs_idx, obs_idx)]
                    cov_mis_obs = cov[np.ix_(mis_idx, obs_idx)]
                    
                    try:
                        inv_cov_obs_obs = np.linalg.pinv(cov_obs_obs)
                        cond_mu = mu_mis + cov_mis_obs @ inv_cov_obs_obs @ (X[i, obs_idx] - mu_obs)
                        
                        X[i, mis_idx] = cond_mu
                    except np.linalg.LinAlgError:
                        X[i, mis_idx] = mu_mis
            
            mu = np.mean(X, axis=0)
            
            if len(numeric_cols) > 0:
                full_cov = np.cov(X, rowvar=False, bias=True)
                
                numeric_idx = [df.columns.get_loc(c) for c in numeric_cols]
                for i, idx_i in enumerate(numeric_idx):
                    for j, idx_j in enumerate(numeric_idx):
                        cov[idx_i, idx_j] = full_cov[idx_i, idx_j]
            
            cov += np.eye(cov.shape[0]) * 1e-6
            
            try:
                log_likelihood = np.sum(multivariate_normal.logpdf(
                    X[:, numeric_idx], 
                    mean=mu[numeric_idx], 
                    cov=cov[np.ix_(numeric_idx, numeric_idx)]
                ))
                log_likelihoods.append(log_likelihood)
                
                if iteration > 0 and abs(log_likelihood - prev_log_likelihood) < tol:
                    print(f"EM算法在迭代 {iteration+1} 次后收敛")
                    break
                    
                prev_log_likelihood = log_likelihood
            except Exception as e:
                print(f"迭代 {iteration+1}: 似然计算错误 - {str(e)}")
        
        if scaler is not None:
            df[numeric_cols] = scaler.inverse_transform(df[numeric_cols])
        
        print("here!")
        filled_df = pd.DataFrame(X, columns=df.columns, index=original_index)
        
        return filled_df

    def em_categorical_imputation(slef, data : pd.DataFrame, max_iter=100, tol=1e-4):
        df = data.copy()
        n_samples, n_features = df.shape
        
        categories = {}
        encoded_data = np.zeros((n_samples, n_features))
        
        for j in range(n_features):
            unique_vals = df.iloc[:, j].dropna().unique()
            categories[j] = {val: idx for idx, val in enumerate(unique_vals)}
            rev_categories = {idx: val for val, idx in categories[j].items()}
            
            for i in range(n_samples):
                val = df.iloc[i, j]
                if pd.isna(val):
                    encoded_data[i, j] = np.nan
                else:
                    encoded_data[i, j] = categories[j].get(val, -1)
        
        n_categories = [len(cat) for cat in categories.values()]
        theta = np.zeros((n_features, max(n_categories)))
        
        for j in range(n_features):
            counts = np.bincount(encoded_data[:, j].astype(int)[~np.isnan(encoded_data[:, j])])
            theta[j, :len(counts)] = counts / counts.sum()
        
        nan_mask = np.isnan(encoded_data)
        
        prev_log_likelihood = -np.inf
        
        for iteration in range(max_iter):
       
            print(iteration)
            for i in range(n_samples):
                missing_cols = np.where(nan_mask[i, :])[0]
                observed_cols = np.where(~nan_mask[i, :])[0]
                
                if len(missing_cols) > 0:
                    joint_probs = np.ones(max(n_categories))
                    
                    for col in missing_cols:
                        probs = theta[col, :n_categories[col]]
                        encoded_data[i, col] = np.argmax(probs)
            
            for j in range(n_features):
                for k in range(n_categories[j]):
                    count = np.sum(encoded_data[:, j] == k)
                    theta[j, k] = count / n_samples
            
            log_likelihood = 0
            for i in range(n_samples):
                sample_ll = 0
                for j in range(n_features):
                    k = int(encoded_data[i, j])
                    sample_ll += np.log(theta[j, k] + 1e-10)
                log_likelihood += sample_ll
            
            if abs(log_likelihood - prev_log_likelihood) < tol:
                break
                
            prev_log_likelihood = log_likelihood
        
        filled_df = pd.DataFrame(index=data.index, columns=data.columns)
        
        for j in range(n_features):
            rev_categories = {idx: val for val, idx in categories[j].items()}
            for i in range(n_samples):
                k = int(encoded_data[i, j])
                filled_df.iloc[i, j] = rev_categories[k]
        
        return filled_df

    def analysis_pipeline(self, data : pd.DataFrame):
        self.get_feature(data)
        


    