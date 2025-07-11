from matplotlib.pyplot import ecdf, isinteractive
import logging
from tqdm import tqdm
from myutils import read_jsonl,write_jsonl
from pathlib import Path
import pdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from typing import Dict

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",  # 日志格式
    filename="unzip.log",
)

class DataProcess:
    def __init__(self):
        self.feature_json_path = 'config/feature.json'
        self.feature_json = read_jsonl(self.feature_json_path)
        self.features_config = self.feature_json['features']
        self.feature_tabel = pd.read_csv('Dataset/FeatureTabel_Ch.csv', index_col=0)
        self.train_data = pd.read_csv('Dataset/diabetic_data_training.csv')
        self.output_dir = Path('Dataset/processed')
        self.dataset_path = Path('Dataset')
        self.ids_mapping = read_jsonl('config/id_mapping.json')
        self.visualization_dir = 'Dataset/processed/visualization'

    def drop_features(self):
        '''
        根据feature_tabel中的keep列，删除train_data中的特征
        '''
        for feature,config in self.features_config.items():
            if config['iskeep']==False:
                self.train_data.drop(columns=[feature], inplace=True)
                print(f'{feature} dropped')

    def reencode(self):
        '''
        读取原始数据，做encoding，确保对于每一个非缺失值，都有对应的encoding_mapping
        '''
        for feature, config in self.features_config.items():
            # 跳过没有在train_data中的特征,identifier,非categorical
            # 获取唯一值（包括nan，nan转为字符串'nan'）
            values = self.train_data[feature]
            unique_values = []
            for v in values.unique():
                if pd.isna(v):
                    print(f'{feature} has nan: {v}')
                else:
                    unique_values.append(str(v))
            # 保证顺序一致
            unique_values = sorted(list(dict.fromkeys(unique_values)))
            config['value_num'] = len(unique_values)
            if feature not in self.train_data.columns or config['category']=='identifier' or config['type']!='categorical':
                continue
            encoding_mapping = {val: idx for idx, val in enumerate(unique_values)}
            config['label_encoding']['unique_values'] = unique_values
            config['label_encoding']['encoding_mapping'] = encoding_mapping
        # 保存
        write_jsonl(self.feature_json, self.feature_json_path)

    def config_features(self):
        '''
        check and fix feature.json, to add config:
            - process: normal, no, special 
            - iskeep: True, False
            - missing: True, False，仅根据FeatureTabel中的Missing Values列判断，但是有的缺失值没有被标记
        '''
        for feature,config in self.features_config.items():
            try:
                if feature in self.feature_tabel.index:
                    if self.feature_tabel.loc[feature]['keep']=='yes':
                        self.features_config[feature]['iskeep'] = True
                        config['process']='normal'
                    elif self.feature_tabel.loc[feature]['keep']=='no':
                        self.features_config[feature]['iskeep'] = False 
                        config['process']='no'
                    else:
                        self.features_config[feature]['iskeep'] = True 
                        config['process']='special'
                else:
                    print(f"Warning: Feature '{feature}' not found in feature table")
                    config['process']='unknown'
                if self.feature_tabel.loc[feature]['Missing Values']=='yes':
                    self.features_config[feature]['missing'] = True
                else:
                    self.features_config[feature]['missing'] = False
            except Exception as e:
                print(f"Error processing feature '{feature}': {e}")
                config['process']='error'

        write_jsonl(self.feature_json,self.feature_json_path)

    def transfer_to_feature_tabel(self):
        '''
        transfer feature.json to FeatureConfigTabel.csv，便于查看
        '''
        feature_tabel = pd.DataFrame(self.features_config).T
        feature_tabel.drop(columns=['label_encoding'], inplace=True)
        feature_tabel.index.name = 'feature_name'
        feature_tabel.to_csv(self.output_dir/'FeatureConfigTabel.csv', index=True)
        print(f'feature.json transferred to {self.output_dir}/FeatureConfigTabel.csv')


    def transfer_all_nan(self):
        '''
        将所有缺失/无意义值转化为缺失值，并在feature.json中记录缺失值种类，得到missing_replaced_train.csv，修改
        - missing_values_num
        - missing_values_p
        - missing_values
        - missing_replace
        - value_num
        - label_encoding
            - unique_values
            - encoding_mapping
        - missing
        '''
        nan_values = self.feature_json['nan_values']
        final_nan = None
        for feature,config in self.features_config.items():
            if config['category']=='identifier' or config['type']!='categorical' or config['iskeep']==False:
                continue
            config['missing_values_num'] = 0
            config['missing_values_p'] = 0
            config['missing_values'] = []
            for idx,data in tqdm(enumerate(self.train_data[feature]),total=len(self.train_data[feature]),desc=f'Processing {feature}'):
                if data in nan_values:
                    try:
                        if isinstance(data,int):
                            data = str(data)
                        elif isinstance(data,str):
                            data = data.strip()
                        if data not in config['missing_values']:
                            config['missing_values'].append(data)
                            config['label_encoding']['unique_values'].remove(data)
                            # config['label_encoding']['encoding_mapping'][final_nan] = config['label_encoding']['encoding_mapping'][data]
                            del config['label_encoding']['encoding_mapping'][data]
                    except:
                        raise ValueError(f"Feature '{feature}' has invalid value: {data}")
                    config['missing_values_num']+=1
                    self.train_data.loc[idx,feature] = final_nan
            config['value_num'] = len(config['label_encoding']['unique_values'])
            if config['missing_values_num']==0:
                config['missing_values_num'] = int(self.train_data[feature].isna().sum())
            config['missing_values_p'] = config['missing_values_num']/len(self.train_data[feature])
            config['missing'] = config['missing_values_num']>0
            config['label_encoding']['unique_values'] = [key for key in config['label_encoding']['encoding_mapping'].keys()]
            config['label_encoding']['encoding_mapping'] = {val:ind for ind,val in enumerate(config['label_encoding']['unique_values'])}
            config['missing_replace'] = final_nan
        try:
            write_jsonl(self.feature_json,self.feature_json_path)
        except Exception as e:
            print(f"Error writing feature.json: {e}")
            pdb.set_trace()
            write_jsonl(self.feature_json,self.feature_json_path)
        print(f'all nan(except id) transferred to {final_nan} in {self.output_dir}/train.csv')

    def transfer_all_nan_for_id(self):
        """
        处理 admission_type_id、discharge_disposition_id、admission_source_id 三个特征：
        1. 根据 ids_mapping 判断含义为未知/空值的 id，将其统一替换为 'missing'
        2. 统计缺失数量与比例
        3. 重新生成 unique_values 与 encoding_mapping，确保二者顺序一致
        """
        id_features = ['admission_type_id', 'discharge_disposition_id', 'admission_source_id']
        nan_meanings = self.feature_json['nan_values']
        final_nan = None

        ids_mapping = self.ids_mapping

        for feature in id_features:
            if feature not in self.features_config:
                print(f"Warning: {feature} not in features_config, skip")
                continue

            config = self.features_config[feature]
            # 获取 id -> 描述 的映射
            id2meaning: dict = ids_mapping.get(feature, {})
            # 找出描述属于 nan_meanings 的 id
            invalid_ids = {
                str(id_) for id_, meaning in id2meaning.items()
                if str(meaning).strip() in nan_meanings
            }

            # 初始化统计字段
            config.setdefault('missing_values', [])
            config['missing_values_num'] = 0

            # 替换无效 id
            col_series = self.train_data[feature]
            replaced_indices = []
            for idx, val in tqdm(col_series.items(),total=len(col_series),desc=f'Processing {feature}'):
                # val 可能为 float/nan 等，统一转 str 方便比较
                val_str = str(int(val)) if pd.notna(val) and str(val).isdigit() else str(val)
                if (pd.isna(val) or val_str in invalid_ids):
                    replaced_indices.append(idx)
                    config['missing']=True
            # 进行替换
            self.train_data.loc[replaced_indices, feature] = final_nan
            config['missing_values_num'] = len(replaced_indices)
            config['missing_values'] = list(invalid_ids) if invalid_ids else []

            # 重新生成 unique_values
            unique_values = [str(v) for v in self.train_data[feature].unique() if pd.notna(v)]

            # 生成新的 encoding_mapping，确保顺序一致
            encoding_mapping = {val: idx for idx, val in enumerate(unique_values)}
            config['label_encoding'] = {
                'unique_values': unique_values,
                'encoding_mapping': encoding_mapping
            }
            config['value_num'] = len(unique_values)
            config['missing_values_p'] = config['missing_values_num'] / len(self.train_data)
            config['missing_replace'] = final_nan

            # 更新回features_config
            self.features_config[feature] = config

        # 保存修改后的 feature.json
        write_jsonl(self.feature_json, self.feature_json_path)
        print(f"Invalid ids in {id_features} have been converted to '{final_nan}' and label encodings updated.")
    
    def recode_train_data(self):
        """将所有保留的分类特征根据 label_encoding 映射为整数编码，并保存 CSV。"""
        self.train_data = pd.read_csv(self.output_dir/'missing_replaced_train.csv')

        for feature, config in self.features_config.items():
            if config.get('category') == 'identifier' or config.get('type') != 'categorical' or config.get('iskeep') is False:
                continue

            enc_map = config['label_encoding']['encoding_mapping']
            # 先统一转成 str 便于匹配
            self.train_data[feature] = self.train_data[feature].astype(str).map(enc_map)
            # 未映射成功的视为缺失，不填充
            # self.train_data[feature] = self.train_data[feature].fillna(-1).astype('Int64')

        # 保存结果
        self.train_data.to_csv(self.output_dir/'recoded_train.csv', index=False)
        print(f'recoded_train.csv saved to {self.output_dir}/recoded_train.csv')

    def check_recoded_data(self):
        check_flag = True
        data = pd.read_csv(self.output_dir/'recoded_train.csv')
        error_features = []
        for feature,config in self.features_config.items():
            if config['category']=='identifier' or config['type']!='categorical' or config['iskeep']==False:
                continue
            print(f'{feature}: {[int(v) for v in sorted(data[feature].unique()) if pd.notna(v)]}')
            vc = data[feature].value_counts()
            missing_num = config.get('missing_values_num',0)
            missing_count = data[feature].isna().sum()
            if missing_num!=missing_count:
                print(f'=========={feature}========== value_num: {len(vc)}, data_num: {len(data[feature])}')
                print(vc)
                print(f'missing_num: {missing_num} / {missing_count}(counted)')
                print(f'Warning: {feature} has {missing_num} missing values, but {missing_count} counted')
                print('--------------------------------')
                check_flag = False
                error_features.append(feature)
                pdb.set_trace()
        if check_flag:
            print('recoded_train.csv check passed')
        else:
            print('recoded_train.csv check failed')
            print(f'error_features: {error_features}')

    def show_feature_config(self):
        for feature,config in self.features_config.items():
            print(f'=========={feature}==========')
            for key,value in config.items():
                if isinstance(value,dict):
                    for k,v in value.items():
                        print(f'{k}: {v}')
                else:
                    print(f'{key}: {value}')
            print('--------------------------------')
            input()

    def save_train_data(self,name):
        self.train_data.to_csv(self.output_dir/f'{name}.csv', index=False)
        print(f'train.csv saved to {self.output_dir}/{name}.csv')

    def visualize_recoded_features(self):
        """
        为重编码后的数据集中的每个特征创建可视化图表。
        从recoded_train.csv读取数据，使用features_config中的特征信息。
        图表保存在visualization_dir目录下。
        """
        # 确保配置和输出目录存在
        os.makedirs(self.visualization_dir, exist_ok=True)
        
        # 读取重编码后的数据
        recoded_data_path = os.path.join(self.output_dir, 'recoded_train.csv')
        if not os.path.exists(recoded_data_path):
            raise FileNotFoundError(f"重编码后的数据文件不存在: {recoded_data_path}. 请先运行recode_train_data()")
        
        df = pd.read_csv(recoded_data_path)
        print(f"读取重编码数据: {recoded_data_path}")
        print("\n数据集中的列名:")
        print(df.columns.tolist())
        print("\n")
        
        # 设置默认的图表样式
        sns.set_style("whitegrid")
        
        # 为每个特征创建直方图
        for feature_name, feature_info in self.features_config.items():
            # 跳过不需要保留的特征
            if not feature_info.get('iskeep', True):
                continue
            feature_id = feature_info.get('feature_id')
            # 检查特征是否在数据集中
            if feature_name not in df.columns:
                print(f"警告: 特征 {feature_name} 在数据集中未找到，跳过")
                continue
                
            # 创建新的图表
            plt.figure(figsize=(12, 6))
            
            # 创建直方图
            if df[feature_name].dtype in ['int64', 'float64']:
                # 数值型特征使用直方图
                sns.histplot(data=df, x=feature_name, bins=30, kde=True)
            else:
                # 分类特征使用计数图
                sns.countplot(data=df, x=feature_name)
                plt.xticks(rotation=45, ha='right')
            
            # 设置标题和标签
            plt.title(f'Distribution of {feature_name}', pad=20)
            plt.xlabel(feature_name)
            plt.ylabel('Count')
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图表
            output_path = os.path.join(self.visualization_dir, f'{feature_id}_{feature_name}.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f'创建特征可视化: {feature_name} -> {output_path}')
    def normalize_data(self):
        '''
        将所有特征归一化到0-1之间，并记录该feature的最大值到feature.json中
        对于离散值和连续值都进行归一化
        '''
        # 读取重编码后的数据
        data_path = os.path.join(self.output_dir, 'recoded_train.csv')
        if not os.path.exists(data_path):
            print(f"重编码后的数据文件不存在: {data_path}. 先进行重编码...")
            self.recode_train_data()
            self.save_train_data('recoded_train')
        
        data = pd.read_csv(data_path)
        print("开始数据归一化...")
        
        # 遍历所有特征
        for feature_name, feature_info in self.features_config.items():
            # 只跳过不需要保留的特征
            if not feature_info.get('iskeep', True):
                continue
                
            print(f"处理特征: {feature_name}")
            
            # 获取特征数据
            if feature_name not in data.columns:
                print(f"警告: 特征 {feature_name} 在数据集中未找到，跳过")
                continue
                
            feature_data = data[feature_name]
            
            # 记录最大值和最小值
            feature_max = float(feature_data.max())
            feature_min = float(feature_data.min())
            
            # 更新feature_config中的信息
            self.features_config[feature_name]['max_value'] = feature_max
            self.features_config[feature_name]['min_value'] = feature_min
            
            # 进行归一化 (x - min) / (max - min)
            if feature_max > feature_min:  # 避免除以0
                data[feature_name] = (feature_data - feature_min) / (feature_max - feature_min)
                # 验证归一化结果
                normalized_max = float(data[feature_name].max())
                normalized_min = float(data[feature_name].min())
                print(f"归一化后范围: [{normalized_min}, {normalized_max}]")
                if not (abs(normalized_max - 1.0) < 1e-6 and abs(normalized_min) < 1e-6):
                    print(f"警告: 特征 {feature_name} 归一化可能不成功")
            else:
                print(f"警告: 特征 {feature_name} 的所有值相同 ({feature_max})，设置为0")
                data[feature_name] = 0
                
        # 保存归一化后的数据
        output_path = os.path.join(self.output_dir, 'normalized_train.csv')
        data.to_csv(output_path, index=False)
        print(f"归一化数据已保存到: {output_path}")
        
        # 保存更新后的feature配置
        with open(self.feature_json_path, 'w', encoding='utf-8') as f:
            json.dump(self.feature_json, f, indent=4, ensure_ascii=False)
        print("特征配置已更新，包含最大值和最小值信息")


def main():
    dp = DataProcess()
    dp.reencode()
    dp.config_features()
    dp.transfer_all_nan()
    dp.transfer_all_nan_for_id()
    # dp.show_feature_config()
    dp.drop_features()
    dp.save_train_data('missing_replaced_train')
    dp.recode_train_data()
    dp.save_train_data('recoded_train')
    dp.check_recoded_data()
    # dp.visualize_recoded_features()
    dp.transfer_to_feature_tabel()
    dp.normalize_data()

if __name__ == '__main__':
    main()