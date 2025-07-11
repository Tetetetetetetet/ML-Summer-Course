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
        self.test_data = None
    
    def load_data(self):
        """
        1. 加载数据
        """
        logging.info("==========load_data==========")
        self.test_data = pd.read_csv('Dataset/diabetic_data_test.csv')
        logging.info(f"测试数据形状: {self.test_data.shape}")

    def config_features(self):
        """
        检查并修复feature.json，添加配置：
            - process: normal, no, special 
            - iskeep: True, False
            - missing: True, False，仅根据FeatureTabel中的Missing Values列判断，但是有的缺失值没有被标记
        """
        logging.info("==========config_features==========")
        for feature, config in self.features_config.items():
            try:
                if feature in self.feature_tabel.index:
                    if self.feature_tabel.loc[feature]['keep'] == 'yes':
                        self.features_config[feature]['iskeep'] = True
                        config['process'] = 'normal'
                    elif self.feature_tabel.loc[feature]['keep'] == 'no':
                        self.features_config[feature]['iskeep'] = False 
                        config['process'] = 'no'
                    else:
                        self.features_config[feature]['iskeep'] = True 
                        config['process'] = 'special'
                else:
                    logging.warning(f"Feature '{feature}' not found in feature table")
                    config['process'] = 'unknown'
                    
                if self.feature_tabel.loc[feature]['Missing Values'] == 'yes':
                    self.features_config[feature]['missing'] = True
                else:
                    self.features_config[feature]['missing'] = False
            except Exception as e:
                logging.error(f"Error processing feature '{feature}': {e}")
                config['process'] = 'error'

        write_jsonl(self.feature_json, self.feature_json_path)
        logging.info("特征配置完成")

    def clean_invalid_data(self):
        """
        有些数据应当被删除：
        - 已死亡的/转入临终关怀的：discharge_disposition_id=[11,19,20,21]/[13,14]
        """
        logging.info("==========clean_invalid_data==========")
        # 是否存在转入临终关怀而再次入院的
        all_hospice = self.test_data[self.test_data['discharge_disposition_id'].isin([13,14])]
        hospice_readmitted = all_hospice[all_hospice['readmitted']!='NO']
        if len(hospice_readmitted) > 0:
            logging.info(f'{len(hospice_readmitted)}/{len(all_hospice)} hospice but readmitted')
        else:
            logging.info('no hospice readmitted')
        # 是否存在已死亡而再次入院的
        all_dead = self.test_data[self.test_data['discharge_disposition_id'].isin([11,19,20,21])]
        dead_readmitted = all_dead[all_dead['readmitted']!='NO']
        if len(dead_readmitted) > 0:
            logging.info(f'{len(dead_readmitted)}/{len(all_dead)} died but readmitted')
        else:
            logging.info('no died readmitted')
        self.test_data = self.test_data[~self.test_data['discharge_disposition_id'].isin([11,19,20,21])]
        logging.info(f'>>>>>>>>>>>after clean invalid data, remained {len(self.test_data)} rows')

    def mark_extream_features(self):
        """
        统计单一值占比极高的特征，并删除
        """
        logging.info("==========mark_extream_features==========")
        for feature, config in self.features_config.items():
            if config['category'] == 'identifier' or config['type'] != 'categorical' or config['iskeep'] == False:
                continue
            if feature not in self.test_data.columns:
                continue
                
            vc = self.test_data[feature].value_counts()
            if vc.iloc[0]/len(self.test_data) > 0.95:
                logging.info(f'{feature} has {vc.iloc[0]}/{len(self.test_data)}({vc.iloc[0]/len(self.test_data)*100:.2f}%) of single value {vc.index[0]}')
                if vc.iloc[0]/len(self.test_data) > 0.98:
                    config['iskeep'] = False
                    logging.info(f'{feature} should drop')
        write_jsonl(self.feature_json, self.feature_json_path)

    def drop_features(self):
        """
        删除特征:
        - iskeep==False
        """
        logging.info("==========drop_features==========")
        drop_features = []
        for feature_name, config in self.features_config.items():
            if not config['iskeep'] and feature_name in self.test_data.columns:
                drop_features.append(feature_name)
                self.test_data = self.test_data.drop(columns=[feature_name])
                logging.info(f"删除特征: {feature_name}")
        logging.info(f"删除特征后，剩余{len(self.test_data.columns)}特征")

        origin_num = len(self.test_data)
        self.test_data = self.test_data.drop(self.test_data[self.test_data['discharge_disposition_id'].isin([11,19,20,21])].index)

        logging.info(f"删除已死亡的数据后，剩余{len(self.test_data)}行, 占比{len(self.test_data)/origin_num*100:.2f}%")

    def transfer_all_nan(self):
        """
        将所有缺失/无意义值转化为缺失值，并在feature.json中记录缺失值种类
        """
        logging.info("==========transfer_all_nan==========")
        nan_values = self.feature_json['nan_values']
        mapping_features = self.ids_mapping.keys()
        
        for feature, config in self.features_config.items():
            if not config['iskeep'] or feature not in self.test_data.columns:
                continue
                
            missing_count = 0
            if config['type'] == 'categorical':
                # 对于分类变量，检查是否在nan_values中
                if feature in mapping_features:
                    # 对于ID特征，先转换为文本进行检查，然后替换缺失值，保持为ID格式
                    temp_values = self.test_data[feature].astype(str).map(self.ids_mapping[feature])
                    mask = temp_values.isin(nan_values)
                    self.test_data[feature] = self.test_data[feature].astype(object)
                    self.test_data.loc[mask, feature] = None
                    missing_count += mask.sum()
                else:
                    # 直接替换nan_values中的值
                    mask = self.test_data[feature].isin(nan_values)
                    self.test_data.loc[mask, feature] = None
                    missing_count += mask.sum()
            
            # 记录缺失值数量
            config['missing_values_num'] = int(missing_count)
            
        write_jsonl(self.feature_json, self.feature_json_path)
        logging.info("所有无意义值已转换为缺失值")

    def analyze_missing_values(self):
        """
        统计缺失值, 将nan_values中的值替换为None
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
                    # 对于ID特征，先转换为文本进行检查，然后替换缺失值，保持为ID格式
                    temp_values = self.test_data[feature].astype(str).map(self.ids_mapping[feature])
                    mask = temp_values.isin(nan_values)
                    self.test_data[feature] = self.test_data[feature].astype(object)
                    self.test_data.loc[mask, feature] = None
                    replace_missing += mask.sum()
                else:
                    replace_missing += (self.test_data[feature].isin(nan_values)).sum()
                    self.test_data[feature] = self.test_data[feature].astype(object)
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
                config['missing_in_test_num'] = int(total_missing)
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
        missing_stats_df.index.name = 'feature_name'
        missing_stats_df.to_csv(self.output_dir / 'test_missing_stats.csv')
        print(f"saved test missing stats to {self.output_dir / 'test_missing_stats.csv'}")
        write_jsonl(self.feature_json,self.feature_json_path)
        
        return missing_stats

    def encode_test_data(self):
        """
        按照feature.json中记录的内容对数据进行编码, 假设已经将nan_values中的值替换为None
        """
        logging.info("==========encode_test_data==========")
        nan_values = self.feature_json['nan_values']
        
        for feature, config in self.features_config.items():
            if not config['iskeep'] or feature not in self.test_data.columns or config['type']!='categorical':
                continue
            
            # 特殊处理诊断特征：使用categorize_diagnosis进行分组
            if feature in ['diag_1', 'diag_2', 'diag_3']:
                logging.info(f"使用categorize_diagnosis处理诊断特征: {feature}")
                self._categorize_diagnosis_feature(feature)
                continue
            
            unseen_values = []
            # 应用编码映射
            if 'label_encoding' in config and 'encoding_mapping' in config['label_encoding']:
                encoding_mapping = config['label_encoding']['encoding_mapping']
            else:
                logging.error(f"无编码信息, 错误: {feature}")
                pdb.set_trace()
                raise ValueError(f"特征 '{feature}' 无编码信息")

            # 对于ID特征，确保转换为字符串格式
            if feature in self.ids_mapping.keys():
                # 将数值转换为字符串，保持ID格式
                self.test_data[feature] = self.test_data[feature].astype(str)
            
            self.test_data[feature] = self.test_data[feature].astype(object)
            train_unique_values = config['label_encoding']['unique_values']
            for val in self.test_data[feature].unique():
                if val and str(val) not in train_unique_values:
                    unseen_values.append(val)
                    logging.error(f"特征 '{feature}' 有值 {val} 不在训练集的编码映射中")
            self.test_data[feature] = self.test_data[feature].astype(str).map(encoding_mapping)
            config['unseen_values'] = unseen_values
            # 检查是否有未映射的值
            nan_num = self.test_data[feature].isna().sum()
            if nan_num > 0 and int(config['missing_in_test_num'])==int(nan_num):
                logging.info(f"特征 '{feature}' 有 {nan_num} 个值[因为缺失]无法映射到编码")
            elif nan_num > 0:
                logging.error(f"无法映射数不等于缺失值数: {nan_num} != {config['missing_in_test_num']}")
                pdb.set_trace()
            # 显示未映射的值
            unmapped_unique = self.test_data[self.test_data[feature].isna()][feature].unique()
            logging.warning(f"未映射的值: {unmapped_unique}")

        
        write_jsonl(self.feature_json,self.feature_json_path)
        self.test_data.to_csv(self.output_dir / 'recoded_test.csv', index=False)

    def _categorize_diagnosis_feature(self, feature_name):
        """
        对单个诊断特征使用categorize_diagnosis进行分组处理（测试集版本）
        
        Args:
            feature_name: 诊断特征名称 (diag_1, diag_2, diag_3)
        """
        # 创建新的分类列名
        new_feature_name = f'new_{feature_name}'
        
        # 复制原始数据
        self.test_data[new_feature_name] = self.test_data[feature_name]
        
        # 保持NaN值为None，不填充为-1
        # self.test_data[new_feature_name] = self.test_data[new_feature_name].fillna(-1)
        
        # 处理包含V或E的编码（外部原因和补充因素）
        # 只处理非NaN的值
        non_null_mask = self.test_data[new_feature_name].notna()
        v_mask = self.test_data.loc[non_null_mask, new_feature_name].astype(str).str.contains('V', na=False)
        e_mask = self.test_data.loc[non_null_mask, new_feature_name].astype(str).str.contains('E', na=False)
        
        self.test_data.loc[non_null_mask & v_mask, new_feature_name] = 0
        self.test_data.loc[non_null_mask & e_mask, new_feature_name] = 0
        
        # 根据ICD-9编码范围进行分类
        for index, row in self.test_data.iterrows():
            code_str = str(row[new_feature_name])
            
            # 跳过NaN值
            if pd.isna(row[new_feature_name]) or code_str == 'nan':
                continue
            
            # 尝试转换为数值进行分类
            try:
                code = float(code_str)
            except ValueError:
                # 如果无法转换为数值，设为类别0（其他疾病）
                self.test_data.loc[index, new_feature_name] = 0
                continue
                
            # 类别1: 循环系统疾病 (390-459, 785)
            if (code >= 390 and code < 460) or (np.floor(code) == 785):
                self.test_data.loc[index, new_feature_name] = 1
            # 类别2: 呼吸系统疾病 (460-519, 786)
            elif (code >= 460 and code < 520) or (np.floor(code) == 786):
                self.test_data.loc[index, new_feature_name] = 2
            # 类别3: 消化系统疾病 (520-579, 787)
            elif (code >= 520 and code < 580) or (np.floor(code) == 787):
                self.test_data.loc[index, new_feature_name] = 3
            # 类别4: 糖尿病 (250)
            elif np.floor(code) == 250:
                self.test_data.loc[index, new_feature_name] = 4
            # 类别5: 损伤和中毒 (800-999)
            elif code >= 800 and code < 1000:
                self.test_data.loc[index, new_feature_name] = 5
            # 类别6: 肌肉骨骼系统疾病 (710-739)
            elif code >= 710 and code < 740:
                self.test_data.loc[index, new_feature_name] = 6
            # 类别7: 泌尿生殖系统疾病 (580-629, 788)
            elif (code >= 580 and code < 630) or (np.floor(code) == 788):
                self.test_data.loc[index, new_feature_name] = 7
            # 类别8: 肿瘤 (140-239)
            elif code >= 140 and code < 240:
                self.test_data.loc[index, new_feature_name] = 8
            # 类别0: 其他疾病
            elif code > 0:
                self.test_data.loc[index, new_feature_name] = 0
        
        # 删除原始诊断列，保留分类后的列
        self.test_data.drop(columns=[feature_name], inplace=True)
        
        # 重命名新列为原始列名
        self.test_data.rename(columns={new_feature_name: feature_name}, inplace=True)
        
        # 应用训练集的编码映射
        config = self.features_config[feature_name]
        encoding_mapping = config['label_encoding']['encoding_mapping']
        self.test_data[feature_name] = self.test_data[feature_name].astype(str).map(encoding_mapping)
        
        logging.info(f"测试集诊断特征 {feature_name} 已分类为9个疾病类别，缺失值保持为None")

    def check_recoded_data(self):
        """
        检查重新编码后的测试数据
        """
        logging.info('==========check_recoded_test.csv==========')
        check_flag = True
        error_features = []
        
        for feature, config in self.features_config.items():
            if config['category'] == 'identifier' or config['type'] != 'categorical' or config['iskeep'] == False:
                continue
            if feature not in self.test_data.columns:
                continue
                
            vc = self.test_data[feature].value_counts()
            missing_num = config.get('missing_in_test_num', 0)
            missing_count = self.test_data[feature].isna().sum()
            
            if missing_num != missing_count:
                logging.warning(f'{feature}: value_num: {len(vc)}, data_num: {len(self.test_data)}')
                logging.warning(f'missing_num: {missing_num} / {missing_count}(counted)')
                logging.warning(f'Warning: {feature} has {missing_num} missing values, but {missing_count} counted')
                check_flag = False
                error_features.append(feature)
        
        if check_flag:
            logging.info('recoded_test.csv check passed')
            logging.info(f'>>>>>>>>>>>after recode, remained {len(self.test_data)} rows, {len(self.test_data.columns)} features')
        else:
            logging.error('recoded_test.csv check failed')
            logging.error(f'error_features: {error_features}')

    def save_train_data(self, name):
        """
        保存测试数据
        """
        self.test_data.to_csv(self.output_dir / f'{name}.csv', index=False)
        logging.info(f'{name}.csv saved to {self.output_dir}/{name}.csv')

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
    handler.config_features()
    handler.clean_invalid_data()
    handler.mark_extream_features()
    handler.drop_features()
    handler.transfer_all_nan()
    missing_stats = handler.analyze_missing_values()
    handler.encode_test_data()
    handler.check_recoded_data()
    handler.save_train_data('recoded_test')
    handler.save_results(missing_stats)
    print(f"测试数据处理完成，处理了 {len(handler.test_data.columns)} 个特征")

if __name__ == '__main__':
    main()