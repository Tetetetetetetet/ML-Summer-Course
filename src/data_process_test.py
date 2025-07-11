import pandas as pd
from myutils import write_jsonl
import pdb
import numpy as np
import json
import os
import logging
from pathlib import Path
from myutils import read_jsonl

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)

class DataProcessTest:
    def __init__(self):
        self.feature_json_path = 'config/feature.json'
        self.feature_json = read_jsonl(self.feature_json_path)
        self.features_config = self.feature_json['features']
        self.feature_tabel = pd.read_csv('Dataset/FeatureTabel_Ch.csv', index_col=0)
        self.output_dir = Path('Dataset/processed/test_processed')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ids_mapping = read_jsonl('config/id_mapping.json')
    
    def load_data(self):
        """
        1. 加载数据
        """
        logging.info("==========load_data==========")
        self.test_data = pd.read_csv('Dataset/diabetic_data_test.csv')
        logging.info(f"测试数据形状: {self.test_data.shape}")

    def drop_features(self):
        """
        删除特征:
        - iskeep==False
        """
        logging.info("==========drop_features==========")
        for feature_name, config in self.features_config.items():
            if not config['iskeep'] and feature_name in self.test_data.columns:
                self.test_data = self.test_data.drop(columns=[feature_name])
                logging.info(f"删除特征: {feature_name}")

        logging.info(f"删除特征后，剩余{len(self.test_data.columns)}特征")

    def analyze_missing_values(self):
        """
        统计缺失值
        """
        logging.info("==========analyze_missing_values==========")
        mapping_features = self.ids_mapping.keys()
        missing_stats = {}
        nan_values = self.feature_json['nan_values']
        
        for feature, config in self.features_config.items():
            if not config['iskeep'] or feature not in self.test_data.columns:
                continue
                
            # 统计原始缺失值
            original_missing = self.test_data[feature].isna().sum()
            
            # 统计需要替换为缺失值的值
            replace_missing = 0
            if config['type'] == 'categorical':
                # 对于分类变量，检查是否在nan_values中
                # 替换为None
                if feature in mapping_features:
                    self.test_data_ = self.test_data.copy()
                    self.test_data_[feature] = self.test_data_[feature].astype(str).map(self.ids_mapping[feature])
                    mask = self.test_data_[feature].isin(nan_values)
                    self.test_data.loc[mask, feature] = None
                    replace_missing += mask.sum()
                else:
                    replace_missing += (self.test_data[feature].isin(nan_values)).sum()
                    self.test_data.loc[self.test_data[feature].isin(nan_values), feature] = None
            
            total_missing = original_missing + replace_missing
            try:
                assert total_missing == self.test_data[feature].isna().sum()
            except:
                pdb.set_trace()
            missing_percentage = (total_missing / len(self.test_data)) * 100
            if total_missing > 0:
                logging.info(f"特征: {feature} 有 {total_missing}({missing_percentage:.2f}%)个缺失值")
                config['missing_in_test'] = True
                # self.test_data[feature].value_counts()
            else:
                config['missing_in_test'] = False
            
            missing_stats[feature] = {
                'original_missing': int(original_missing),
                'replace_missing': int(replace_missing),
                'total_missing': int(total_missing),
                'missing_percentage': float(missing_percentage)
            }
        
        # 保存缺失值统计
        missing_stats_df = pd.DataFrame.from_dict(missing_stats, orient='index')
        missing_stats_df.to_csv(self.output_dir / 'test_missing_stats.csv')
        print(f"saved test missing stats to {self.output_dir / 'test_missing_stats.csv'}")
        write_jsonl(self.feature_json,self.feature_json_path)
        
        return missing_stats

    def encode_test_data(self):
        """
        按照feature.json中记录的内容对数据进行编码
        """
        logging.info("==========encode_test_data==========")
        nan_values = self.feature_json['nan_values']
        
        for feature, config in self.features_config.items():
            if not config['iskeep'] or feature not in self.test_data.columns:
                continue
                
            logging.info(f"处理特征: {feature}")
            
            # 替换缺失值
            if config['type'] == 'categorical':
                # 将nan_values中的值替换为None
                for nan_val in nan_values:
                    self.test_data.loc[self.test_data[feature] == nan_val, feature] = None
            
            # 应用编码映射
            if 'label_encoding' in config and 'encoding_mapping' in config['label_encoding']:
                encoding_mapping = config['label_encoding']['encoding_mapping']
                
                # 创建反向映射（从字符串到编码）
                reverse_mapping = {str(k): v for k, v in encoding_mapping.items()}
                
                # 应用编码
                self.test_data[feature] = self.test_data[feature].astype(str).map(reverse_mapping)
                
                # 检查是否有未映射的值
                unmapped_values = self.test_data[feature].isna().sum()
                if unmapped_values > 0:
                    logging.warning(f"特征 '{feature}' 有 {unmapped_values} 个值无法映射到编码")
                    
                    # 显示未映射的值
                    unmapped_unique = self.test_data[self.test_data[feature].isna()][feature].unique()
                    logging.warning(f"未映射的值: {unmapped_unique}")

    def save_results(self, missing_stats):
        """
        保存处理结果
        """
        logging.info("==========save_results==========")
        
        # 保存处理后的测试数据
        self.test_data.to_csv(self.output_dir / 'processed_test_data.csv', index=False)
        
        # 生成处理报告
        report = {
            'test_data_info': {
                'original_shape': self.test_data.shape,
                'features_processed': len([f for f, c in self.features_config.items() if c['iskeep'] and f in self.test_data.columns])
            },
            'missing_value_summary': {
                'total_features_with_missing': len([f for f, stats in missing_stats.items() if stats['total_missing'] > 0]),
                'total_missing_values': sum(stats['total_missing'] for stats in missing_stats.values()),
                'average_missing_percentage': float(np.mean([stats['missing_percentage'] for stats in missing_stats.values()]))
            },
            'encoding_summary': {
                'categorical_features_encoded': len([f for f, c in self.features_config.items() 
                                                   if c['iskeep'] and c['type'] == 'categorical' and f in self.test_data.columns])
            }
        }
        
        # 保存报告
        with open(self.output_dir / 'test_processing_report.json', 'w') as f:
            json.dump(report, f, indent=4)
        
        logging.info(f"测试数据处理完成，结果保存在: {self.output_dir}")
        logging.info(f"处理了 {report['test_data_info']['features_processed']} 个特征")
        logging.info(f"总共 {report['missing_value_summary']['total_missing_values']} 个缺失值")
        
def main():
    handler = DataProcessTest()
    handler.load_data()
    handler.drop_features()
    missing_stats = handler.analyze_missing_values()
    return
    handler.encode_test_data()
    handler.save_results(missing_stats)
    print(f"测试数据处理完成，处理了 {len(handler.test_data.columns)} 个特征")

if __name__ == '__main__':
    main()