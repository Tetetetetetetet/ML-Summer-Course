import pandas as pd
import shutil
import numpy as np
import os
import logging
import json
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE, chi2
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from argparse import ArgumentParser
from imblearn.over_sampling import SMOTE

# 兼容不同环境的导入方式
try:
    from myutils import read_jsonl
except ImportError:
    try:
        from .myutils import read_jsonl
    except ImportError:
        try:
            from src.myutils import read_jsonl
        except ImportError:
            raise ImportError("myutils不可用，请检查myutils.py是否存在")
# 兼容不同环境的导入方式
try:
    from .resnet_model import ResNet
except ImportError:
    try:
        from resnet_model import ResNet
    except ImportError:
        try:
            from src.resnet_model import ResNet
        except ImportError:
            # 如果ResNet不可用，创建一个占位符类
            class ResNet:
                def __init__(self, **kwargs):
                    raise ImportError("ResNet模型不可用，请检查TensorFlow安装")

# 设置日志
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
warnings.filterwarnings('ignore')

class DataFit:
    def __init__(self, args=None):
        """
        初始化DataFit类
        
        Args:

        """
        self.output_dir = 'output/'
        self.cover_old_result = args.cover_old_result
        if not self.cover_old_result:
            try_id = 1
            while True:
                self.results_dir = os.path.join(self.output_dir, f"{args.exp_name}-{try_id}")
                if not os.path.exists(self.results_dir):
                    break
                try_id += 1
        else:
            self.results_dir = os.path.join(self.output_dir, args.exp_name)
        os.makedirs(self.results_dir, exist_ok=True)
        shutil.copy('train.sh', self.results_dir)
        self.dataset_dir = 'Dataset/processed/train_processed'
        self.mode = args.mode
        self.exp_name = args.exp_name
        self.train_data = None
        self.test_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.isoversample = args.isoversample
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = None
        self.results = {}
        self.feature_json = read_jsonl('config/feature.json')
        self.feature_config = self.feature_json['features']
        self.k = args.k
        if self.k != 'all':
            self.k = int(self.k)
        self.feature_selectors = {
            'f_classif': SelectKBest(score_func=f_classif, k=self.k),
            'chi2': SelectKBest(score_func=chi2, k=self.k),
        }
        if args.feature_selector!="None":
            self.feature_selector = (args.feature_selector, self.feature_selectors[args.feature_selector])
        else:
            self.feature_selector = None
        self.mode2dataset = {
            'normal': {'train': 'improved_logistic_imputed/improved_logistic_imputed_train_final.csv','test': 'improved_logistic_imputed/improved_logistic_imputed_test_final.csv'},
            'selected': {'train': 'improved_logistic_imputed/improved_logistic_imputed_train_final_selected.csv','test': 'improved_logistic_imputed/improved_logistic_imputed_test_final_selected.csv'},
            '2class': {'train': 'improved_logistic_imputed/improved_logistic_imputed_train_final_2class.csv','test': 'improved_logistic_imputed/improved_logistic_imputed_test_final_2class.csv'},
            'network_data': {'train': 'network_train.csv','test': 'network_test.csv'},
            'nan_as_newclass': {'train': 'nan_as_newclass_train.csv','test': 'nan_as_newclass_test.csv'},
            'one_hot': {'train':'','test':''}
        }
        self.models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1)
        }
        
        # 添加ResNet模型（如果可用）
        # 根据特征选择结果确定ResNet的特征数量
        if self.k == 'all':
            resnet_n = None  # 使用所有特征
        else:
            resnet_n = self.k
        
        self.models['ResNet'] = ResNet(
            n=resnet_n,
            epochs=args.resnet_epochs,
            batch_size=64,
            gpu_batch_size=args.resnet_gpu_batch_size,
            validation_split=0.2,
            random_state=42,
            need_train=True,
        )
        self.models['ResNet'].set_mode(self.mode)
        if args.model != 'all':
            self.models = {args.model: self.models[args.model]}
        
        logging.info(f"DataFit初始化完成，mode: {self.mode}")
    
    def load_data(self):
        """
        加载逻辑回归填充后的完整数据集
        """
        logging.info("==========load_data==========")
        
        try:
            # 加载训练集和测试集
            if self.mode in self.mode2dataset:
                train_path = os.path.join(self.dataset_dir, self.mode2dataset[self.mode]['train'])
                test_path = os.path.join(self.dataset_dir, self.mode2dataset[self.mode]['test'])
            else:
                raise ValueError(f"Invalid mode: {self.mode}")
            
            self.train_data = pd.read_csv(train_path)
            self.test_data = pd.read_csv(test_path)
            
            logging.info(f"训练集加载完成: {self.train_data.shape}, from {train_path}")
            logging.info(f"测试集加载完成: {self.test_data.shape}, from {test_path}")
            
            # 检查数据完整性
            logging.info(f"训练集缺失值: {self.train_data.isna().sum().sum()}")
            logging.info(f"测试集缺失值: {self.test_data.isna().sum().sum()}")
            
            # 显示目标变量分布
            if 'readmitted' in self.train_data.columns:
                target_dist = self.train_data['readmitted'].value_counts()
                logging.info(f"目标变量分布:\n{target_dist}")
            
        except Exception as e:
            logging.error(f"数据加载失败: {e}")
            raise
    
    def preprocess_data(self):
        """
        数据预处理：特征工程、缩放等
        """
        logging.info("==========preprocess_data==========")
        
        # 分离特征和目标变量
        target_col = 'readmitted'
        if target_col not in self.train_data.columns:
            logging.error(f"目标变量 '{target_col}' 不存在")
            return
        
        assert target_col in self.train_data.columns, f"目标变量 '{target_col}' 不存在"
        assert target_col in self.test_data.columns, f"目标变量 '{target_col}' 不存在"
        
        # 准备训练数据
        X_train_full = self.train_data.drop(columns=[target_col])
        y_train_full = self.train_data[target_col]
        
        # 准备测试数据
        X_test_full = self.test_data.drop(columns=[target_col])
        y_test_full = self.test_data[target_col]
        
        logging.info(f"特征数量: {len(X_train_full.columns)} 个")
        
        # 特征选择 - 保留全部特征
        logging.info("进行特征选择...")
        if self.feature_selector:
            selector_name,selector = self.feature_selector
            X_train_selected = selector.fit_transform(X_train_full, y_train_full)
            X_test_selected = selector.transform(X_test_full)
            # 记录特征分数
            feature_scores = pd.DataFrame(selector.scores_, index=X_train_full.columns, columns=['score'])
            feature_scores.to_csv(os.path.join(self.results_dir, f'feature_scores_{selector_name}.csv'), index=True)
            logging.info(f"开始特征选择，保留{len(selected_features)}个特征，使用{selector_name}方法,\n特征分数保存到{os.path.join(self.results_dir, f'feature_scores_{selector_name}.csv')}\n选择特征: {selected_features}")
            # 画出特征分数图
            plt.figure(figsize=(10, 6))
            sns.barplot(x=feature_scores['score'], y=feature_scores.index)
            plt.title(f'Feature Scores for {selector_name}')
            plt.xlabel('Score')
            plt.ylabel('Features')
            plt.savefig(os.path.join(self.results_dir, f'feature_scores_{selector_name}.png'))
            plt.close()
            selected_features = X_train_full.columns[selector.get_support()].tolist()
        else:
            X_train_selected = X_train_full
            X_test_selected = X_test_full
            selected_features = X_train_full.columns.tolist()
                # 数据标准化
        X_train_selected = self.scaler.fit_transform(X_train_selected)
        X_test_selected = self.scaler.transform(X_test_selected)
            
        if self.isoversample:
            smt = SMOTE()
            X_train_selected, y_train_full = smt.fit_resample(X_train_selected, y_train_full)
        
        # 转换为DataFrame
        self.X_train = pd.DataFrame(X_train_selected, columns=selected_features)
        self.X_test = pd.DataFrame(X_test_selected, columns=selected_features)
        self.y_train = y_train_full
        self.y_test = y_test_full
        
        logging.info(f"预处理完成 - 训练集: {self.X_train.shape}, 测试集: {self.X_test.shape}")
    
    def train_models(self):
        """
        训练多个机器学习模型
        """
        logging.info("==========train_models==========")
        
        # 定义模型 - 训练多个模型
        models = self.models.copy()  # 创建副本避免迭代时修改
        
        # 训练模型
        failed_models = []
        for name, model in models.items():
            logging.info(f"训练模型: {name}")
            try:
                model.fit(self.X_train, self.y_train)
                self.models[name] = model
                logging.info(f"模型 {name} 训练完成")
            except Exception as e:
                logging.error(f"模型 {name} 训练失败: {e}")
                failed_models.append(name)
                # 从模型字典中移除失败的模型
                if name in self.models:
                    del self.models[name]
        
        if failed_models:
            logging.warning(f"以下模型训练失败，将被跳过: {failed_models}")
    
    def evaluate_models(self):
        """
        评估模型性能
        """
        logging.info("==========evaluate_models==========")
        
        results = {}
        
        for name, model in self.models.items():
            logging.info(f"评估模型: {name}")
            
            # 预测
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test) if hasattr(model, 'predict_proba') else None
            
            # 计算指标
            accuracy = accuracy_score(self.y_test, y_pred)
            
            # 计算F1-macro
            f1_macro = f1_score(self.y_test, y_pred, average='macro')
            
            results[name] = {
                'accuracy': accuracy,
                'f1_macro': f1_macro,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            # 如果有概率预测，计算AUC
            if y_pred_proba is not None:
                auc_result = self.calculate_auc(self.y_test, y_pred_proba, name)
                results[name]['auc'] = auc_result
            
            # 打印总体表现
            logging.info(f"模型 {name} 总体表现:")
            logging.info(f"  准确率: {accuracy:.4f}")
            logging.info(f"  F1-Macro: {f1_macro:.4f}")
            if results[name].get('auc'):
                logging.info(f"  AUC: {results[name]['auc']:.4f}")
            
            # 保存详细报告
            report = classification_report(self.y_test, y_pred, output_dict=True)
            
            '''
            # 打印各类别详细指标
            logging.info(f"模型 {name} 各类别表现:")
            for class_name in ['0', '1', '2']:
                if class_name in report:
                    class_report = report[class_name]
                    support = class_report['support']
                    precision = class_report['precision']
                    recall = class_report['recall']
                    f1 = class_report['f1-score']
                    logging.info(f"  类别{class_name}: precision={precision:.3f}, recall={recall:.3f}, f1-score={f1:.3f}, support={support}")
            '''
            results[name]['f1_macro'] = report['macro avg']['f1-score']
            results[name]['classification_report'] = report
        
        self.results = results
        
        # 检查是否有成功训练的模型
        if not results:
            logging.error("没有成功训练的模型，无法进行评估")
            return
        
        # 找到最佳模型（综合考虑准确率和F1-macro）
        best_model_name = max(results.keys(), key=lambda x: results[x]['f1_macro'])
        self.best_model = self.models[best_model_name]
        self.best_model_name = best_model_name
        
        best_result = results[best_model_name]
        logging.info(f"最佳模型: {best_model_name}")
        logging.info(f"  准确率: {best_result['accuracy']:.4f}")
        logging.info(f"  F1-Macro: {best_result['f1_macro']:.4f}")
        if best_result.get('auc'):
            logging.info(f"  AUC: {best_result['auc']:.4f}")
    
    def calculate_auc(self, y_true, y_pred_proba, model_name):
        """
        计算AUC的通用方法，支持二分类和多分类
        """
        try:
            # 检查目标变量的类别数量
            unique_classes = np.unique(y_true)
            n_classes = len(unique_classes)
            
            logging.info(f"目标变量类别: {unique_classes}, 数量: {n_classes}")
            
            if n_classes == 2:
                # 二分类情况 - 取正类的概率
                if y_pred_proba.shape[1] == 2:
                    # 如果概率矩阵是2列，取第1列（正类概率）
                    auc = roc_auc_score(y_true, y_pred_proba[:, 1])
                else:
                    # 如果概率矩阵是1列，直接使用
                    auc = roc_auc_score(y_true, y_pred_proba)
                logging.info(f"模型 {model_name} 使用二分类AUC")
                
            elif n_classes > 2:
                # 多分类情况 - 使用one-vs-rest方法
                auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
                logging.info(f"模型 {model_name} 使用多分类AUC (one-vs-rest, macro average)")
                
                # 也可以计算每个类别的AUC
                auc_per_class = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average=None)
                logging.info(f"模型 {model_name} 各类别AUC: {dict(zip(unique_classes, auc_per_class))}")
                
            else:
                logging.warning(f"模型 {model_name} 目标变量类别数量异常: {n_classes}")
                return None
                
            return auc
            
        except Exception as e:
            logging.warning(f"模型 {model_name} 计算AUC失败: {e}")
            return None
    
    def hyperparameter_tuning(self, model_name='RandomForest'):
        """
        对指定模型进行超参数调优
        """
        logging.info(f"==========hyperparameter_tuning for {model_name}==========")
        
        if model_name not in self.models:
            logging.error(f"模型 {model_name} 不存在")
            return
        
        # 定义参数网格
        param_grids = {
            'RandomForest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            },
            'GradientBoosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'LogisticRegression': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],  # 更细致的正则化强度
                'penalty': ['l1', 'l2', 'elasticnet', None],  # 添加弹性网络和无正则化
                'solver': ['liblinear', 'saga', 'lbfgs', 'newton-cg'],  # 更多求解器
                'max_iter': [100, 200, 500, 1000],  # 最大迭代次数
                'tol': [1e-4, 1e-3, 1e-2],  # 收敛容差
                'class_weight': [None, 'balanced'],  # 类别权重
                'multi_class': ['ovr', 'multinomial']  # 多分类策略
            }
        }
        
        if model_name not in param_grids:
            logging.warning(f"模型 {model_name} 没有预定义的参数网格")
            return
        
        # 对于LogisticRegression，创建兼容的参数组合
        if model_name == 'LogisticRegression':
            param_grid = self._create_logistic_param_grid()
        else:
            param_grid = param_grids[model_name]
        
        # 定义评估指标
        scoring = {
            'accuracy': 'accuracy',
            'precision_macro': 'precision_macro',
            'recall_macro': 'recall_macro',
            'f1_macro': 'f1_macro',
            'roc_auc_ovr': 'roc_auc_ovr'
        }
        
        # 网格搜索
        grid_search = GridSearchCV(
            self.models[model_name],
            param_grid,
            cv=5,
            scoring=scoring,
            refit='f1_macro',  # 使用F1-macro作为主要指标
            n_jobs=-1,
            verbose=1,
            error_score=0
        )
        
        try:
            grid_search.fit(self.X_train, self.y_train)
            
            # 更新最佳模型
            self.models[f'{model_name}_tuned'] = grid_search.best_estimator_
            self.best_model = grid_search.best_estimator_
            self.best_model_name = f'{model_name}_tuned'
            
            logging.info(f"最佳参数: {grid_search.best_params_}")
            logging.info(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")
            
            # 显示所有评估指标
            logging.info("交叉验证结果:")
            for metric, score in grid_search.cv_results_['mean_test_score'].items():
                logging.info(f"  {metric}: {score:.4f}")
            
            # 重新评估
            self.evaluate_models()
            
        except Exception as e:
            logging.error(f"超参数调优失败: {e}")
            logging.info("尝试使用简化的参数网格...")
            self._fallback_hyperparameter_tuning(model_name)
    
    def _create_logistic_param_grid(self):
        """
        创建LogisticRegression的兼容参数网格
        """
        param_grid = []
        
        # 基础参数组合
        base_params = {
            'C': [0.1, 1, 10],
            'max_iter': [500],  # 增加迭代次数避免收敛问题
            'tol': [1e-4],
            'class_weight': [None, 'balanced']
        }
        
        # 不同求解器的参数组合（基于测试结果优化）
        solver_params = [
            # liblinear - 支持l1和l2，表现较好
            {'solver': ['liblinear'], 'penalty': ['l1', 'l2'], 'multi_class': ['ovr']},
            # lbfgs - 只支持l2，稳定
            {'solver': ['lbfgs'], 'penalty': ['l2'], 'multi_class': ['ovr']},
            # saga - 支持多种正则化，但需要更多参数
            {'solver': ['saga'], 'penalty': ['l1', 'l2'], 'multi_class': ['ovr']},
            # newton-cg - 只支持l2
            {'solver': ['newton-cg'], 'penalty': ['l2'], 'multi_class': ['ovr']}
        ]
        
        for solver_param in solver_params:
            for C in base_params['C']:
                for max_iter in base_params['max_iter']:
                    for tol in base_params['tol']:
                        for class_weight in base_params['class_weight']:
                            for solver in solver_param['solver']:
                                for penalty in solver_param['penalty']:
                                    for multi_class in solver_param['multi_class']:
                                        # 跳过不兼容的组合
                                        if penalty == 'l1' and solver not in ['liblinear', 'saga']:
                                            continue
                                        
                                        param_combination = {
                                            'C': C,
                                            'solver': solver,
                                            'penalty': penalty,
                                            'max_iter': max_iter,
                                            'tol': tol,
                                            'class_weight': class_weight,
                                            'multi_class': multi_class
                                        }
                                        param_grid.append(param_combination)

        logging.info(f"生成了 {len(param_grid)} 个兼容的参数组合")
        return param_grid
    
    def _fallback_hyperparameter_tuning(self, model_name):
        """
        简化版超参数调优（备用方案）
        """
        logging.info(f"使用简化版超参数调优: {model_name}")
        
        fallback_params = {
            'LogisticRegression': {
                'C': [0.1, 1, 10],
                'penalty': ['l2'],
                'solver': ['lbfgs'],
                'max_iter': [200],
                'class_weight': [None, 'balanced']
            }
        }
        
        if model_name not in fallback_params:
            return
        
        grid_search = GridSearchCV(
            self.models[model_name],
            fallback_params[model_name],
            cv=5,
            scoring='f1_macro',
            n_jobs=-1,
            verbose=1
        )
        
        try:
            grid_search.fit(self.X_train, self.y_train)
            
            self.models[f'{model_name}_tuned'] = grid_search.best_estimator_
            self.best_model = grid_search.best_estimator_
            self.best_model_name = f'{model_name}_tuned'
            
            logging.info(f"简化版最佳参数: {grid_search.best_params_}")
            logging.info(f"简化版最佳分数: {grid_search.best_score_:.4f}")
            
            self.evaluate_models()
            
        except Exception as e:
            logging.error(f"简化版超参数调优也失败: {e}")
    
    def feature_importance_analysis(self):
        """
        分析特征重要性
        """
        logging.info("==========feature_importance_analysis==========")
        
        if self.best_model is None:
            logging.error("没有可用的最佳模型")
            return
        
        # ResNet模型不支持特征重要性分析
        if self.best_model_name == 'ResNet':
            logging.info("ResNet模型不支持特征重要性分析，跳过此步骤")
            return
        
        # 获取特征重要性
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
        elif hasattr(self.best_model, 'coef_'):
            importances = np.abs(self.best_model.coef_[0])
        else:
            logging.warning("模型不支持特征重要性分析")
            return
        
        # 创建特征重要性DataFrame
        feature_importance_df = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # 保存特征重要性
        importance_path = os.path.join(self.results_dir, 'feature_importance.csv')
        feature_importance_df.to_csv(importance_path, index=False)
        
        # 绘制特征重要性图
        plt.figure(figsize=(12, 8))
        top_features = feature_importance_df.head(15)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top 15 Feature Importance - {self.best_model_name}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        # 保存图片
        importance_plot_path = os.path.join(self.results_dir, 'feature_importance.png')
        plt.savefig(importance_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"特征重要性分析完成，结果保存在: {importance_path}")
        logging.info(f"前5个重要特征: {feature_importance_df.head()['feature'].tolist()}")
    
    def generate_predictions(self, output_file=None):
        """
        生成预测结果
        """
        logging.info("==========generate_predictions==========")
        
        if self.best_model is None:
            logging.error("没有可用的最佳模型")
            return
        
        # 对测试集进行预测
        predictions = self.best_model.predict(self.X_test)
        probabilities = self.best_model.predict_proba(self.X_test) if hasattr(self.best_model, 'predict_proba') else None
        
        # 创建预测结果DataFrame
        results_df = pd.DataFrame({
            'predicted_readmitted': predictions
        })
        
        if probabilities is not None:
            # 处理多分类概率矩阵
            if len(probabilities.shape) == 2 and probabilities.shape[1] > 1:
                # 多分类情况 - 保存每个类别的概率
                for i in range(probabilities.shape[1]):
                    results_df[f'probability_class_{i}'] = probabilities[:, i]
                logging.info(f"保存了 {probabilities.shape[1]} 个类别的概率")
            else:
                # 二分类情况
                results_df['prediction_probability'] = probabilities
        
        # 保存预测结果
        if output_file is None:
            output_file = os.path.join(self.results_dir, f'predictions.csv')
        
        results_df.to_csv(output_file, index=False)
        logging.info(f"预测结果保存到: {output_file}")
        
        return results_df
    
    def save_model(self, model_path=None):
        """
        保存最佳模型
        """
        logging.info("==========save_model==========")
        
        if self.best_model is None:
            logging.error("没有可用的最佳模型")
            return
        
        if model_path is None:
            if self.best_model_name == 'ResNet':
                model_path = os.path.join(self.results_dir, 'best_model.h5')
            else:
                model_path = os.path.join(self.results_dir, 'best_model.pkl')
        
        if self.best_model_name == 'ResNet':
            # ResNet模型已经在其内部保存，这里只需要记录信息
            self.best_model._save_model(model_path)
        else:
            # 传统机器学习模型保存为.pkl格式
            import pickle
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': self.best_model,
                    'scaler': self.scaler,
                    'feature_selector': self.feature_selector,
                    'feature_names': self.X_train.columns.tolist(),
                    'model_name': self.best_model_name
                }, f)
            logging.info(f"模型保存到: {model_path}")
    
    def generate_report(self):
        """
        生成完整的建模报告
        """
        logging.info("==========generate_report==========")
        
        report = {
            'best_model': self.best_model_name,
            'best_model_f1_macro': float(self.results[self.best_model_name]['f1_macro']),
            'training_data_shape': self.X_train.shape,
            'selected_features': self.X_train.columns.tolist(),
            'test_data_shape': self.X_test.shape,
            'model_performance': {},
            'feature_importance': {},
            'timestamp': datetime.now().isoformat()
        }
        if self.best_model_name == 'ResNet':
            report['best_model_hash'] = self.best_model.hash
        
        # 添加模型性能
        for name, result in self.results.items():
            report['model_performance'][name] = {
                'accuracy': result['accuracy'],
                'auc': result.get('auc'),
                'f1_macro': result.get('f1_macro'),
                'classification_report': result.get('classification_report', {})
            }
        
        # 保存报告
        report_path = os.path.join(self.results_dir, 'modeling_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4, default=str)
        
        logging.info(f"建模报告保存到: {report_path}")
        
        # 打印总结
        logging.info("==========建模总结==========")
        logging.info(f"实验名称: {self.exp_name}")
        logging.info(f"数据集模式: {self.mode}")
        logging.info(f"是否过采样: {self.isoversample}")
        logging.info(f"最佳模型: {self.best_model_name}")
        logging.info(f"最佳准确率: {self.results[self.best_model_name]['accuracy']:.4f}")
        logging.info(f"最佳F1-Macro: {self.results[self.best_model_name]['f1_macro']:.4f}")
        if self.results[self.best_model_name].get('auc'):
            logging.info(f"最佳AUC: {self.results[self.best_model_name]['auc']:.4f}")
    
    def run_complete_pipeline(self):
        """
        运行完整的建模流程
        """
        logging.info("==========开始完整建模流程==========")
        
        try:
            # 1. 加载数据
            self.load_data()
            
            # 2. 数据预处理
            self.preprocess_data()
            
            # 3. 训练模型
            self.train_models()
            
            # 4. 评估模型
            self.evaluate_models()
            
            # 5. 超参数调优（可选）
            self.hyperparameter_tuning('LogisticRegression')
            
            # 6. 特征重要性分析
            self.feature_importance_analysis()
            
            # 7. 生成预测
            self.generate_predictions()
            
            # 8. 保存模型
            self.save_model()
            
            # 9. 生成报告
            self.generate_report()
            
            logging.info("==========建模流程完成==========")
            
        except Exception as e:
            logging.error(f"建模流程失败: {e}")
            raise e

def main():
    """
    主函数
    """
    parser = ArgumentParser()
    parser.add_argument('--mode', type=str, default='normal', help='数据集模式')
    parser.add_argument('-s','--isoversample', default=False, help='过采样',action='store_true')
    parser.add_argument('-e','--exp_name',type=str,default='normal_exp',help='实验名称')
    parser.add_argument('-k','--k',type=str,default='all',help='特征选择保留的特征数量("all" or int)')
    parser.add_argument('--model',type=str,default='LogisticRegression',help='模型名称 or "all"')
    parser.add_argument('--feature_selector',type=str,default='chi2',help='特征选择方法, can be "chi2" or "f_classif"')
    parser.add_argument('--resnet_gpu_batch_size',type=int,default=512,help='GPU批次大小')
    parser.add_argument('--resnet_epochs',type=int,default=200,help='ResNet训练轮数')
    parser.add_argument('--cover_old_result',type=bool,default=False,help='是否覆盖旧的实验结果')
    args = parser.parse_args()
    data_fit = DataFit(args=args)
    data_fit.run_complete_pipeline()


if __name__ == "__main__":
    main()