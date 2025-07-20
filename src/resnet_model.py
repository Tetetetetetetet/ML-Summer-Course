import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score
import logging
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import hashlib
import pickle

class ResNet:
    """
    ResNet-like 全连接网络模型类
    完全兼容data_fit.py的接口，可以直接添加到self.models中使用
    """
    
    def __init__(self, feature_cols=None, n=None, epochs=200, batch_size=64, 
                 validation_split=0.2, class_weight=None, random_state=42,
                 need_train=True, model_save_dir='output/resnet_models',
                 use_network_order=False, use_network_data=False):
        """
        初始化ResNet模型
        
        Args:
            feature_cols: 指定要使用的特征列编号（1-based 列表或 range）
            n: 当 feature_cols=None 时生效；使用前 n 个特征
            epochs: 训练轮数
            batch_size: 批次大小
            validation_split: 验证集比例
            class_weight: 类别权重
            random_state: 随机种子
            need_train: 是否需要训练，True时训练并保存，False时尝试加载已有模型
            model_save_dir: 模型保存目录
            use_network_order: 是否使用与Network版本相同的特征顺序
            use_network_data: 是否直接使用Network版本的数据集
        """
        self.feature_cols = feature_cols
        self.n = n
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.class_weight = class_weight
        self.random_state = random_state
        self.need_train = need_train
        self.model_save_dir = model_save_dir
        self.use_network_order = use_network_order
        self.use_network_data = use_network_data
        
        # 设置随机种子
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        
        # 初始化组件
        self.model = None
        self.scaler = StandardScaler()
        self.actual_feature_cols = None
        self.history = None
        self.is_fitted = False
        self.input_dim = None
        self.mode = None  # 数据集模式，由data_fit.py设置
        
        # 创建模型保存目录
        os.makedirs(self.model_save_dir, exist_ok=True)
        
        logging.info(f"ResNet模型初始化完成，need_train={need_train}, use_network_data={use_network_data}")
    
    def _build_model(self, input_dim):
        """
        构建ResNet-like模型
        
        Args:
            input_dim: 输入特征维度
        """
        self.input_dim = input_dim
        
        def res_block(x, units, dropout_rate):
            """ResNet残差块"""
            shortcut = x
            x = tf.keras.layers.Dense(units, activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(dropout_rate)(x)
            x = tf.keras.layers.Dense(units)(x)
            x = tf.keras.layers.Add()([shortcut, x])
            x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            return x
        
        # 构建网络
        inputs = tf.keras.layers.Input(shape=(input_dim,))
        x = tf.keras.layers.Dense(256, activation='relu')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        
        # 添加3个残差块
        for _ in range(3):
            x = res_block(x, 256, 0.4)
        
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
        
        self.model = tf.keras.Model(inputs, outputs)
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logging.info(f"ResNet模型构建完成，输入维度: {input_dim}")
    
    def _get_model_params_hash(self):
        """
        获取模型参数的哈希值，用于标识模型
        """
        params = {
            'feature_cols': self.feature_cols,
            'n': self.n,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'validation_split': self.validation_split,
            'class_weight': self.class_weight,
            'random_state': self.random_state,
            'mode': self.mode
        }
        params_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(params_str.encode()).hexdigest()
    
    def _get_model_params_for_comparison(self):
        """
        获取用于比较的模型参数（不包含need_train等运行时参数）
        """
        return {
            'feature_cols': self.feature_cols,
            'n': self.n,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'validation_split': self.validation_split,
            'class_weight': self.class_weight,
            'random_state': self.random_state,
            'mode': self.mode
        }
    
    def _find_existing_model(self):
        """
        查找是否存在参数一致的已训练模型
        
        Returns:
            model_path: 模型路径，如果找到的话
        """
        if self.need_train:
            return None
        
        params_hash = self._get_model_params_hash()
        
        # 遍历模型保存目录
        for model_dir in os.listdir(self.model_save_dir):
            model_path = os.path.join(self.model_save_dir, model_dir)
            if not os.path.isdir(model_path):
                continue
            
            # 检查参数文件
            params_file = os.path.join(model_path, 'model_params.json')
            if not os.path.exists(params_file):
                continue
            
            try:
                with open(params_file, 'r') as f:
                    saved_params = json.load(f)
                
                # 比较参数
                current_params = self._get_model_params_for_comparison()
                
                # 从保存的参数中提取用于比较的参数
                saved_params_for_comparison = {k: v for k, v in saved_params.items() 
                                            if k in current_params}
                
                if saved_params_for_comparison == current_params:
                    model_file = os.path.join(model_path, 'model.h5')
                    if os.path.exists(model_file):
                        logging.info(f"找到参数一致的已训练模型: {model_path}")
                        return model_path
            except Exception as e:
                logging.warning(f"读取模型参数失败: {e}")
                continue
        
        return None
    
    def _save_model(self, model_dir):
        """
        保存模型和参数
        
        Args:
            model_dir: 模型保存目录
        """
        os.makedirs(model_dir, exist_ok=True)
        
        # 保存模型
        model_path = os.path.join(model_dir, 'model.h5')
        self.model.save(model_path)
        
        # 保存参数
        params = self._get_model_params_for_comparison()
        # 添加训练时生成的参数
        params.update({
            'input_dim': self.input_dim,
            'actual_feature_cols': self.actual_feature_cols
        })
        
        params_file = os.path.join(model_dir, 'model_params.json')
        with open(params_file, 'w') as f:
            json.dump(params, f, indent=2)
        
        # 保存scaler
        scaler_file = os.path.join(model_dir, 'scaler.pkl')
        with open(scaler_file, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # 保存训练历史
        if self.history is not None:
            history_file = os.path.join(model_dir, 'history.json')
            with open(history_file, 'w') as f:
                json.dump(self.history.history, f, indent=2)
        
        logging.info(f"模型已保存到: {model_dir}")
    
    def _load_model(self, model_dir):
        """
        加载模型和参数
        
        Args:
            model_dir: 模型目录
        """
        # 加载模型
        model_path = os.path.join(model_dir, 'model.h5')
        self.model = tf.keras.models.load_model(model_path)
        
        # 加载参数
        params_file = os.path.join(model_dir, 'model_params.json')
        with open(params_file, 'r') as f:
            params = json.load(f)
        
        self.input_dim = params['input_dim']
        self.actual_feature_cols = params['actual_feature_cols']
        
        # 加载scaler
        scaler_file = os.path.join(model_dir, 'scaler.pkl')
        with open(scaler_file, 'rb') as f:
            self.scaler = pickle.load(f)
        
        self.is_fitted = True
        logging.info(f"模型已从 {model_dir} 加载")
    
    def _prepare_data(self, X, is_training=True):
        """
        准备数据：标准化
        
        Args:
            X: 输入特征数据
            is_training: 是否为训练模式
        
        Returns:
            X_scaled: 标准化后的特征数据
        """
        if is_training:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def _calculate_class_weights(self, y):
        """
        计算类别权重
        """
        if self.class_weight is not None:
            return self.class_weight
        
        # 自动计算类别权重
        class_counts = np.bincount(y)
        total_samples = len(y)
        n_classes = len(class_counts)
        weights = dict(enumerate(total_samples / (n_classes * class_counts)))
        
        logging.info(f"类别权重: {weights}")
        return weights
    
    def set_mode(self, mode):
        """
        设置数据集模式（由data_fit.py调用）
        
        Args:
            mode: 数据集模式
        """
        self.mode = mode
    
    def _load_network_data(self):
        """
        加载Network版本的数据集
        
        Returns:
            X_train, y_train, X_test, y_test: 训练和测试数据
        """
        try:
            # 加载Network版本的数据
            train_df = pd.read_csv('Network/Data/train.csv')
            test_df = pd.read_csv('Network/Data/test.csv')
            
            # 提取特征和目标变量
            X_train = train_df.iloc[:, :28].astype('float32').values
            y_train = train_df['readmitted'].astype('int32').values
            
            X_test = test_df.iloc[:, :28].astype('float32').values
            y_test = test_df['readmitted'].astype('int32').values
            
            # 标准化
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
            
            logging.info(f"成功加载Network版本数据: 训练集{X_train.shape}, 测试集{X_test.shape}")
            return X_train, y_train, X_test, y_test
            
        except Exception as e:
            logging.error(f"加载Network版本数据失败: {e}")
            raise
    
    def fit(self, X, y):
        """
        训练模型（兼容sklearn接口）
        
        Args:
            X: 特征数据（DataFrame或numpy数组）
            y: 目标变量
        
        Returns:
            self: 返回自身，兼容sklearn接口
        """
        logging.info("==========ResNet模型训练开始==========")
        
        # 如果使用Network数据，直接加载
        if self.use_network_data:
            logging.info("使用Network版本数据集")
            X_train, y_train, X_test, y_test = self._load_network_data()
            # 保存测试数据供后续使用
            self.X_test_network = X_test
            self.y_test_network = y_test
            X, y = X_train, y_train
        else:
            # 确保X是numpy数组
            if hasattr(X, 'values'):
                X = X.values
            if hasattr(y, 'values'):
                y = y.values
        
        # 检查是否需要训练
        if not self.need_train:
            existing_model = self._find_existing_model()
            if existing_model is not None:
                logging.info("找到已训练的模型，直接加载")
                self._load_model(existing_model)
                logging.info("==========ResNet模型加载完成==========")
                return self
            else:
                logging.warning("未找到已训练的模型，将进行训练")
        
        # 确定特征列
        if self.feature_cols is None:
            if self.n is None:
                # 使用所有特征
                self.actual_feature_cols = list(range(X.shape[1]))
            else:
                # 使用前n个特征
                if X.shape[1] < self.n:
                    logging.error(f"特征数量({X.shape[1]})小于指定数量({self.n})")
                    raise ValueError(f"特征数量({X.shape[1]})小于指定数量({self.n})")
                
                if self.use_network_order and self.n == 28:
                    # 使用与Network版本相同的特征顺序
                    logging.info("使用Network版本的特征顺序")
                    # 检查数据是否包含目标变量
                    if X.shape[1] == 29:  # 包含目标变量
                        # 在集成版本中，目标变量在第26列，需要排除它
                        # 使用前25列 + 第27-28列（跳过第26列的目标变量）
                        self.actual_feature_cols = list(range(25)) + [26, 27]
                        logging.info("检测到数据包含目标变量，使用前25列+第27-28列作为特征")
                    else:
                        self.actual_feature_cols = list(range(28))
                else:
                    self.actual_feature_cols = list(range(self.n))
        else:
            self.actual_feature_cols = list(self.feature_cols)
        
        # 检查特征维度
        if len(self.actual_feature_cols) > 256:  # 第一层维度
            logging.error(f"特征数量({len(self.actual_feature_cols)})大于第一层维度(256)，将只使用前256个特征")
            self.actual_feature_cols = self.actual_feature_cols[:256]
        
        # 选择特征
        try:
            logging.info(f"选择特征，特征索引: {self.actual_feature_cols}")
            logging.info(f"输入数据形状: {X.shape}")
            X_selected = X[:, self.actual_feature_cols]
            logging.info(f"选择后数据形状: {X_selected.shape}")
        except Exception as e:
            logging.error(f"特征选择失败: {e}")
            logging.error(f"特征索引: {self.actual_feature_cols}")
            logging.error(f"输入数据形状: {X.shape}")
            raise
        
        # 构建模型
        self._build_model(X_selected.shape[1])
        
        # 准备数据
        X_scaled = self._prepare_data(X_selected, is_training=True)
        
        # 计算类别权重
        class_weights = self._calculate_class_weights(y)
        
        # 训练模型
        self.history = self.model.fit(
            X_scaled, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            class_weight=class_weights,
            verbose=2
        )
        
        self.is_fitted = True
        
        # 保存模型
        if self.need_train:
            params_hash = self._get_model_params_hash()
            model_dir = os.path.join(self.model_save_dir, f"resnet_{params_hash}")
            self._save_model(model_dir)
        
        logging.info("==========ResNet模型训练完成==========")
        return self
    
    def predict(self, X):
        """
        预测类别（兼容sklearn接口）
        
        Args:
            X: 特征数据（DataFrame或numpy数组）
        
        Returns:
            predictions: 预测类别
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 如果使用Network数据，使用保存的测试数据
        if self.use_network_data and hasattr(self, 'X_test_network'):
            X = self.X_test_network
        else:
            # 确保X是numpy数组
            if hasattr(X, 'values'):
                X = X.values
        
        # 选择特征
        try:
            X_selected = X[:, self.actual_feature_cols]
        except Exception as e:
            logging.error(f"预测时特征选择失败: {e}")
            logging.error(f"特征索引: {self.actual_feature_cols}")
            logging.error(f"输入数据形状: {X.shape}")
            raise
        
        # 准备数据
        X_scaled = self._prepare_data(X_selected, is_training=False)
        
        # 预测概率
        probabilities = self.model.predict(X_scaled)
        
        # 预测类别
        predictions = np.argmax(probabilities, axis=1)
        
        return predictions
    
    def predict_proba(self, X):
        """
        预测概率（兼容sklearn接口）
        
        Args:
            X: 特征数据（DataFrame或numpy数组）
        
        Returns:
            probabilities: 预测概率
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 如果使用Network数据，使用保存的测试数据
        if self.use_network_data and hasattr(self, 'X_test_network'):
            X = self.X_test_network
        else:
            # 确保X是numpy数组
            if hasattr(X, 'values'):
                X = X.values
        
        # 选择特征
        try:
            X_selected = X[:, self.actual_feature_cols]
        except Exception as e:
            logging.error(f"预测概率时特征选择失败: {e}")
            logging.error(f"特征索引: {self.actual_feature_cols}")
            logging.error(f"输入数据形状: {X.shape}")
            raise
        
        # 准备数据
        X_scaled = self._prepare_data(X_selected, is_training=False)
        
        # 预测概率
        probabilities = self.model.predict(X_scaled)
        
        return probabilities
    
    def get_params(self, deep=True):
        """
        获取模型参数（兼容sklearn接口）
        """
        return {
            'feature_cols': self.feature_cols,
            'n': self.n,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'validation_split': self.validation_split,
            'class_weight': self.class_weight,
            'random_state': self.random_state,
            'need_train': self.need_train,
            'model_save_dir': self.model_save_dir,
            'use_network_order': self.use_network_order,
            'use_network_data': self.use_network_data
        }
    
    def set_params(self, **params):
        """
        设置模型参数（兼容sklearn接口）
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self


def create_resnet_model(feature_cols=None, n=None, **kwargs):
    """
    创建ResNet模型的工厂函数
    
    Args:
        feature_cols: 特征列
        n: 特征数量
        **kwargs: 其他参数
    
    Returns:
        ResNet实例
    """
    return ResNet(feature_cols=feature_cols, n=n, **kwargs) 