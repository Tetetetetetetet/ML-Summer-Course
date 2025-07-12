import pandas as pd
import numpy as np
import logging
import os
from pathlib import Path
from data_missing import MissingDataHandler
import json

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)

class LogisticImputationPipeline:
    def __init__(self):
        self.train_output_dir = Path('Dataset/processed/train_processed')
        self.test_output_dir = Path('Dataset/processed/test_processed')
        self.handler = MissingDataHandler()
        
    def load_recoded_data(self):
        """
        加载recoded后的训练集和测试集数据
        """
        logging.info("==========加载recoded数据==========")
        
        # 加载训练集
        train_path = self.train_output_dir / 'recoded_train.csv'
        if not train_path.exists():
            raise FileNotFoundError(f"训练集文件不存在: {train_path}")
        self.train_data = pd.read_csv(train_path)
        logging.info(f"训练集加载完成: {self.train_data.shape}")
        
        # 加载测试集
        test_path = self.test_output_dir / 'recoded_test.csv'
        if not test_path.exists():
            raise FileNotFoundError(f"测试集文件不存在: {test_path}")
        self.test_data = pd.read_csv(test_path)
        logging.info(f"测试集加载完成: {self.test_data.shape}")
        
        # 检查训练集和测试集的列是否一致
        train_cols = set(self.train_data.columns)
        test_cols = set(self.test_data.columns)
        
        if train_cols != test_cols:
            missing_in_test = train_cols - test_cols
            missing_in_train = test_cols - train_cols
            if missing_in_test:
                logging.warning(f"测试集中缺少的列: {missing_in_test}")
            if missing_in_train:
                logging.warning(f"训练集中缺少的列: {missing_in_train}")
        
        # 确保训练集和测试集有相同的列
        common_cols = list(train_cols.intersection(test_cols))
        self.train_data = self.train_data[common_cols]
        self.test_data = self.test_data[common_cols]
        
        logging.info(f"训练集和测试集列对齐完成: {len(common_cols)} 个共同特征")
        
    def analyze_missing_values(self):
        """
        分析训练集和测试集的缺失值情况
        """
        logging.info("==========缺失值分析==========")
        
        # 分析训练集缺失值
        train_missing = {}
        for col in self.train_data.columns:
            missing_count = self.train_data[col].isna().sum()
            if missing_count > 0:
                missing_pct = (missing_count / len(self.train_data)) * 100
                train_missing[col] = {'count': missing_count, 'percentage': missing_pct}
                logging.info(f"训练集 {col}: {missing_count} 缺失值 ({missing_pct:.2f}%)")
        
        # 分析测试集缺失值
        test_missing = {}
        for col in self.test_data.columns:
            missing_count = self.test_data[col].isna().sum()
            if missing_count > 0:
                missing_pct = (missing_count / len(self.test_data)) * 100
                test_missing[col] = {'count': missing_count, 'percentage': missing_pct}
                logging.info(f"测试集 {col}: {missing_count} 缺失值 ({missing_pct:.2f}%)")
        
        # 保存缺失值分析结果
        missing_analysis = {
            'train_missing': train_missing,
            'test_missing': test_missing,
            'total_train_missing': sum(info['count'] for info in train_missing.values()),
            'total_test_missing': sum(info['count'] for info in test_missing.values()),
            'features_with_missing': list(set(list(train_missing.keys()) + list(test_missing.keys())))
        }
        
        # 保存分析结果
        analysis_path = self.train_output_dir / 'logistic_imputed' / 'missing_analysis.json'
        analysis_path.parent.mkdir(exist_ok=True)
        with open(analysis_path, 'w') as f:
            json.dump(missing_analysis, f, indent=4, default=str)
        
        logging.info(f"缺失值分析完成，结果保存在: {analysis_path}")
        return missing_analysis
    
    def run_logistic_imputation(self):
        """
        运行逻辑回归缺失值填充流程
        """
        logging.info("==========开始逻辑回归缺失值填充==========")
        
        # 1. 对训练集进行逻辑回归填充
        logging.info("步骤1: 对训练集进行逻辑回归填充")
        self.handler.train_data = self.train_data
        imputed_train, model_info = self.handler.logistic_regression_impute(self.train_data)
        
        # 2. 对测试集使用训练好的模型进行填充
        logging.info("步骤2: 对测试集使用训练好的模型进行填充")
        imputed_test = self.handler.logistic_regression_impute_test(self.test_data)
        
        # 3. 验证填充结果
        logging.info("步骤3: 验证填充结果")
        self.verify_imputation_results(imputed_train, imputed_test)
        
        # 4. 保存最终结果
        logging.info("步骤4: 保存最终结果")
        self.save_final_results(imputed_train, imputed_test)
        
        return imputed_train, imputed_test
    
    def verify_imputation_results(self, imputed_train, imputed_test):
        """
        验证填充结果
        """
        logging.info("==========验证填充结果==========")
        
        # 检查是否还有缺失值
        train_missing_after = imputed_train.isna().sum().sum()
        test_missing_after = imputed_test.isna().sum().sum()
        
        logging.info(f"填充后训练集缺失值: {train_missing_after}")
        logging.info(f"填充后测试集缺失值: {test_missing_after}")
        
        if train_missing_after > 0:
            logging.warning(f"训练集仍有 {train_missing_after} 个缺失值")
            missing_cols = imputed_train.columns[imputed_train.isna().any()].tolist()
            logging.warning(f"仍有缺失值的列: {missing_cols}")
        
        if test_missing_after > 0:
            logging.warning(f"测试集仍有 {test_missing_after} 个缺失值")
            missing_cols = imputed_test.columns[imputed_test.isna().any()].tolist()
            logging.warning(f"仍有缺失值的列: {missing_cols}")
        
        # 检查数据形状
        logging.info(f"填充后训练集形状: {imputed_train.shape}")
        logging.info(f"填充后测试集形状: {imputed_test.shape}")
        
        # 检查数据类型
        logging.info("训练集数据类型:")
        for col in imputed_train.columns:
            dtype = imputed_train[col].dtype
            unique_count = imputed_train[col].nunique()
            logging.info(f"  {col}: {dtype}, {unique_count} 个唯一值")
        
        logging.info("测试集数据类型:")
        for col in imputed_test.columns:
            dtype = imputed_test[col].dtype
            unique_count = imputed_test[col].nunique()
            logging.info(f"  {col}: {dtype}, {unique_count} 个唯一值")
    
    def save_final_results(self, imputed_train, imputed_test):
        """
        保存最终结果
        """
        logging.info("==========保存最终结果==========")
        
        # 创建输出目录
        output_dir = self.train_output_dir / 'logistic_imputed'
        output_dir.mkdir(exist_ok=True)
        
        # 保存填充后的数据
        train_output_path = output_dir / 'logistic_imputed_train_final.csv'
        test_output_path = output_dir / 'logistic_imputed_test_final.csv'
        
        imputed_train.to_csv(train_output_path, index=False)
        imputed_test.to_csv(test_output_path, index=False)
        
        logging.info(f"训练集保存到: {train_output_path}")
        logging.info(f"测试集保存到: {test_output_path}")
        
        # 生成处理报告
        report = {
            'imputation_method': 'logistic_regression',
            'train_data_info': {
                'original_shape': self.train_data.shape,
                'imputed_shape': imputed_train.shape,
                'original_missing': self.train_data.isna().sum().sum(),
                'imputed_missing': imputed_train.isna().sum().sum()
            },
            'test_data_info': {
                'original_shape': self.test_data.shape,
                'imputed_shape': imputed_test.shape,
                'original_missing': self.test_data.isna().sum().sum(),
                'imputed_missing': imputed_test.isna().sum().sum()
            },
            'features_processed': len(self.train_data.columns),
            'processing_timestamp': pd.Timestamp.now().isoformat()
        }
        
        # 保存报告
        report_path = output_dir / 'imputation_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4, default=str)
        
        logging.info(f"处理报告保存到: {report_path}")
        
        # 打印总结
        logging.info("==========逻辑回归填充完成==========")
        logging.info(f"训练集: {self.train_data.shape} -> {imputed_train.shape}")
        logging.info(f"测试集: {self.test_data.shape} -> {imputed_test.shape}")
        logging.info(f"训练集缺失值: {self.train_data.isna().sum().sum()} -> {imputed_train.isna().sum().sum()}")
        logging.info(f"测试集缺失值: {self.test_data.isna().sum().sum()} -> {imputed_test.isna().sum().sum()}")

def main():
    """
    主函数：运行完整的逻辑回归缺失值填充流程
    """
    logging.info("==========逻辑回归缺失值填充流程开始==========")
    
    # 创建pipeline实例
    pipeline = LogisticImputationPipeline()
    
    try:
        # 1. 加载recoded数据
        pipeline.load_recoded_data()
        
        # 2. 分析缺失值
        missing_analysis = pipeline.analyze_missing_values()
        
        # 3. 运行逻辑回归填充
        imputed_train, imputed_test = pipeline.run_logistic_imputation()
        
        logging.info("==========逻辑回归缺失值填充流程完成==========")
        
    except Exception as e:
        logging.error(f"流程执行失败: {e}")
        raise

if __name__ == '__main__':
    main() 