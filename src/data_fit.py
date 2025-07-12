import pandas as pd
import numpy as np
import os
import logging
import json
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from argparse import ArgumentParser
from imblearn.over_sampling import SMOTE

# 设置日志
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
warnings.filterwarnings('ignore')

class DataFit:
    def __init__(self, data_source='logistic_imputed',mode='normal',isoversample=False):
        """
        初始化DataFit类
        
        Args:
            data_source: 数据源类型 ('logistic_imputed', 'knn_imputed', 'mean_imputed', 'median_imputed')
        """
        self.output_dir = 'Dataset/processed/train_processed'
        self.data_source = data_source
        self.mode = mode
        self.train_data = None
        self.test_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.isoversample = isoversample
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.best_model = None
        self.best_model_name = None
        self.results = {}
        self.mode2dataset = {
            'normal': {'train': 'logistic_imputed/logistic_imputed_train_final.csv','test': 'logistic_imputed/logistic_imputed_test_final.csv'},
            '2class': {'train': 'logistic_imputed/logistic_imputed_train_final_2class.csv','test': 'logistic_imputed/logistic_imputed_test_final_2class.csv'},
        }
        
        # 创建结果保存目录
        self.results_dir = os.path.join(self.output_dir, 'modeling_results' if not self.isoversample else 'modeling_oversample_results')
        os.makedirs(self.results_dir, exist_ok=True)
        
        logging.info(f"DataFit初始化完成，数据源: {data_source}")
    
    def load_data(self):
        """
        加载逻辑回归填充后的完整数据集
        """
        logging.info("==========load_data==========")
        
        try:
            # 加载训练集和测试集
            if self.mode in self.mode2dataset:
                train_path = os.path.join(self.output_dir, self.mode2dataset[self.mode]['train'])
                test_path = os.path.join(self.output_dir, self.mode2dataset[self.mode]['test'])
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
        数据预处理：特征工程、编码、缩放等
        """
        logging.info("==========preprocess_data==========")
        
        # 分离特征和目标变量
        target_col = 'readmitted'
        if target_col not in self.train_data.columns:
            logging.error(f"目标变量 '{target_col}' 不存在")
            return
        
        # 准备训练数据
        X_train_full = self.train_data.drop(columns=[target_col])
        y_train_full = self.train_data[target_col]
        
        # 准备测试数据（如果有目标变量）
        if target_col in self.test_data.columns:
            X_test_full = self.test_data.drop(columns=[target_col])
            y_test_full = self.test_data[target_col]
        else:
            X_test_full = self.test_data.copy()
            y_test_full = None
        
        # 处理分类变量
        categorical_features = []
        numeric_features = []
        
        for col in X_train_full.columns:
            if X_train_full[col].dtype == 'object' or X_train_full[col].nunique() < 10:
                categorical_features.append(col)
            else:
                numeric_features.append(col)
        
        logging.info(f"分类特征: {len(categorical_features)} 个")
        logging.info(f"数值特征: {len(numeric_features)} 个")
        
        # 对分类变量进行标签编码
        label_encoders = {}
        X_train_encoded = X_train_full.copy()
        X_test_encoded = X_test_full.copy()
        
        for col in categorical_features:
            le = LabelEncoder()
            # 先全部转成字符串
            X_train_encoded[col] = X_train_encoded[col].astype(str)
            X_test_encoded[col] = X_test_encoded[col].astype(str)
            # 训练集fit
            le.fit(X_train_encoded[col])
            # 测试集中的新类别全部替换成训练集众数
            train_classes = set(le.classes_)
            test_classes = set(X_test_encoded[col].unique())
            unseen = test_classes - train_classes
            if unseen:
                logging.warning(f"特征 '{col}' 在测试集中有新的类别: {unseen}")
                most_common = X_train_encoded[col].mode().iloc[0]
                X_test_encoded[col] = X_test_encoded[col].replace(list(unseen), most_common)
            # 再transform
            X_train_encoded[col] = le.transform(X_train_encoded[col])
            X_test_encoded[col] = le.transform(X_test_encoded[col])
            label_encoders[col] = le
        
        # 特征选择 - 保留全部特征
        logging.info("进行特征选择...")
        self.feature_selector = SelectKBest(score_func=f_classif, k='all')
        logging.info(f"开始特征选择，保留全部 {len(X_train_encoded.columns)} 个特征...")
        X_train_selected = self.feature_selector.fit_transform(X_train_encoded, y_train_full)
        X_test_selected = self.feature_selector.transform(X_test_encoded)
        
        # 获取选中的特征名称
        selected_features = X_train_encoded.columns[self.feature_selector.get_support()].tolist()
        logging.info(f"特征选择完成，选中的特征: {selected_features}")
        
        # 数据标准化
        X_train_scaled = self.scaler.fit_transform(X_train_selected)
        X_test_scaled = self.scaler.transform(X_test_selected)
        
        
        if self.isoversample:
            smt = SMOTE()
            X_train_scaled, y_train_full = smt.fit_resample(X_train_scaled, y_train_full)
        
        # 转换为DataFrame
        self.X_train = pd.DataFrame(X_train_scaled, columns=selected_features)
        self.X_test = pd.DataFrame(X_test_scaled, columns=selected_features)
        self.y_train = y_train_full
        self.y_test = y_test_full
        
        logging.info(f"预处理完成 - 训练集: {self.X_train.shape}, 测试集: {self.X_test.shape}")
    
    def train_models(self):
        """
        训练多个机器学习模型
        """
        logging.info("==========train_models==========")
        
        # 定义模型 - 训练多个模型
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1)
        }
        
        # 训练模型
        for name, model in models.items():
            logging.info(f"训练模型: {name}")
            try:
                model.fit(self.X_train, self.y_train)
                self.models[name] = model
                logging.info(f"模型 {name} 训练完成")
            except Exception as e:
                logging.error(f"模型 {name} 训练失败: {e}")
    
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
            y_pred_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # 计算指标
            accuracy = accuracy_score(self.y_test, y_pred)
            results[name] = {
                'accuracy': accuracy,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            # 如果有概率预测，计算AUC
            if y_pred_proba is not None:
                try:
                    auc = roc_auc_score(self.y_test, y_pred_proba)
                    results[name]['auc'] = auc
                except:
                    results[name]['auc'] = None
            
            # 打印分类报告
            logging.info(f"模型 {name} 准确率: {accuracy:.4f}")
            if results[name].get('auc'):
                logging.info(f"模型 {name} AUC: {results[name]['auc']:.4f}")
            
            # 保存详细报告
            report = classification_report(self.y_test, y_pred, output_dict=True)
            results[name]['classification_report'] = report
        
        self.results = results
        
        # 找到最佳模型
        best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        self.best_model = self.models[best_model_name]
        self.best_model_name = best_model_name
        
        logging.info(f"最佳模型: {best_model_name} (准确率: {results[best_model_name]['accuracy']:.4f})")
    
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
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        }
        
        if model_name not in param_grids:
            logging.warning(f"模型 {model_name} 没有预定义的参数网格")
            return
        
        # 网格搜索
        grid_search = GridSearchCV(
            self.models[model_name],
            param_grids[model_name],
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        # 更新最佳模型
        self.models[f'{model_name}_tuned'] = grid_search.best_estimator_
        self.best_model = grid_search.best_estimator_
        self.best_model_name = f'{model_name}_tuned'
        
        logging.info(f"最佳参数: {grid_search.best_params_}")
        logging.info(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")
        
        # 重新评估
        self.evaluate_models()
    
    def feature_importance_analysis(self):
        """
        分析特征重要性
        """
        logging.info("==========feature_importance_analysis==========")
        
        if self.best_model is None:
            logging.error("没有可用的最佳模型")
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
        probabilities = self.best_model.predict_proba(self.X_test)[:, 1] if hasattr(self.best_model, 'predict_proba') else None
        
        # 创建预测结果DataFrame
        results_df = pd.DataFrame({
            'predicted_readmitted': predictions
        })
        
        if probabilities is not None:
            results_df['prediction_probability'] = probabilities
        
        # 保存预测结果
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.results_dir, f'predictions_{timestamp}.csv')
        
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
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(self.results_dir, f'best_model_{timestamp}.pkl')
        
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
            'data_source': self.data_source,
            'best_model': self.best_model_name,
            'training_data_shape': self.X_train.shape,
            'test_data_shape': self.X_test.shape,
            'model_performance': {},
            'feature_importance': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # 添加模型性能
        for name, result in self.results.items():
            report['model_performance'][name] = {
                'accuracy': result['accuracy'],
                'auc': result.get('auc'),
                'classification_report': result.get('classification_report', {})
            }
        
        # 保存报告
        report_path = os.path.join(self.results_dir, 'modeling_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4, default=str)
        
        logging.info(f"建模报告保存到: {report_path}")
        
        # 打印总结
        logging.info("==========建模总结==========")
        logging.info(f"数据源: {self.data_source}")
        logging.info(f"最佳模型: {self.best_model_name}")
        logging.info(f"最佳准确率: {self.results[self.best_model_name]['accuracy']:.4f}")
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
            # self.hyperparameter_tuning('RandomForest')
            
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
    parser.add_argument('-m','--mode', type=str, default='normal', help='数据集模式')
    parser.add_argument('-s','--isoversample', type=bool, default=False, help='数据集模式',action='store_true')
    args = parser.parse_args()
    data_fit = DataFit(data_source='logistic_imputed',mode=args.mode,isoversample=args.isoversample)
    data_fit.run_complete_pipeline()


if __name__ == "__main__":
    main()