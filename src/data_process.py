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
)

def categorize_diagnosis(new,old):
  '''
  The diagnosis columns are converted from icd9 codes to one of the 9 categories.
  The function creates a new column which consists of one of the 10 categories of disease.
  Nan values are considered as a new category
  '''
  #copying old column into new column
  data[new] = data[old]
  #filling NaN values with -1 category
  data[new] = data[new].fillna(-1)
  #if the code contians V or E then it is considered as category 0.
  data.loc[data[new].str.contains('V',na=False), [new]] = 0
  data.loc[data[new].str.contains('E',na=False), [new]] = 0
  #converting string column to float.
  data[new] = data[new].astype(float)
  #iterating through all the rows of the dataframe
  for index, row in data.iterrows():
      #checking if the code of the row belong to category 1,
      if (row[new] >= 390 and row[new] < 460) or (np.floor(row[new]) == 785):
          #assigning category 1 to the new column of the row.
          data.loc[index, new] = 1
      #checking if the code of the row belong to category 2,
      elif (row[new] >= 460 and row[new] < 520) or (np.floor(row[new]) == 786):
          #assigning category 2 to the new column of the row.
          data.loc[index, new] = 2
      elif (row[new] >= 520 and row[new] < 580) or (np.floor(row['new_diag1']) == 787):
          data.loc[index, new] = 3
      elif (np.floor(row[new]) == 250):
          data.loc[index, new] = 4
      elif (row[new] >= 800 and row[new] < 1000):
          data.loc[index, new] = 5
      elif (row[new] >= 710 and row[new] < 740):
          data.loc[index, new] = 6
      elif (row[new] >= 580 and row[new] < 630) or (np.floor(row[new]) == 788):
          data.loc[index, new] = 7
      elif (row[new] >= 140 and row[new] < 240):
          data.loc[index, new] = 8
      elif row[new] > 0 :
        #if the code of the row does not belong to any category then it is given 0.
          data.loc[index, new] = 0

class DataProcess:
    def __init__(self):
        self.feature_json_path = 'config/feature.json'
        self.feature_json = read_jsonl(self.feature_json_path)
        self.features_config = self.feature_json['features']
        self.feature_tabel = pd.read_csv('Dataset/FeatureTabel_Ch.csv', index_col=0)
        self.train_data = pd.read_csv('Dataset/diabetic_data_training.csv')
        self.output_dir = Path('Dataset/processed/train_processed')
        self.dataset_path = Path('Dataset')
        self.ids_mapping = read_jsonl('config/id_mapping.json')
        self.visualization_dir = 'Dataset/processed/visualization'

    def load_train_data(self,mode:str='origin'):
        if mode=='origin':
            self.train_data = pd.read_csv('Dataset/diabetic_data_training.csv')
        elif mode=='missing_replaced':
            self.train_data = pd.read_csv('Dataset/missing_replaced_train.csv')
        elif mode=='recoded':
            self.train_data = pd.read_csv('Dataset/recoded_train.csv')
        elif mode=='normalized':
            self.train_data = pd.read_csv('Dataset/normalized_train.csv')

    def exp_json(self):
        dic = {"exist":True,"name":"mine"}
        write_jsonl(dic,'config/exp.json')

    def clean_invalid_data(self):
        '''
        有些数据应当被删除：
        - 已死亡的/转入临终关怀的：discharge_disposition_id=[11,19,20,21]/[13,14]
        '''
        # 是否存在转入临终关怀而再次入院的
        all_hospice = self.train_data[self.train_data['discharge_disposition_id'].isin([13,14])]
        hospice_readmitted = all_hospice[all_hospice['readmitted']!='NO']
        if len(hospice_readmitted)>0:
            logging.info(f'{len(hospice_readmitted)}/{len(all_hospice)} hospice but readmitted')
        else:
            logging.info('no hospice readmitted')
        # 是否存在已死亡而再次入院的
        all_dead = self.train_data[self.train_data['discharge_disposition_id'].isin([11,19,20,21])]
        dead_readmitted = all_dead[all_dead['readmitted']!='NO']
        if len(dead_readmitted)>0:
            logging.info(f'{len(dead_readmitted)}/{len(all_dead)} died but readmitted')
        else:
            logging.info('no died readmitted')
        self.train_data = self.train_data[~self.train_data['discharge_disposition_id'].isin([11,19,20,21])]
        logging.info(f'>>>>>>>>>>>after clean invalid data, remained {len(self.train_data)} rows')

    def mark_extream_features(self):
        '''
        统计单一值占比极高的特征，并删除
        '''
        for feature,config in self.features_config.items():
            if config['category']=='identifier' or config['type']!='categorical' or config['iskeep']==False:
                continue
            vc = self.train_data[feature].value_counts()
            if vc.iloc[0]/len(self.train_data)>0.95:
                logging.info(f'{feature} has {vc.iloc[0]}/{len(self.train_data)}({vc.iloc[0]/len(self.train_data)*100:.2f}%) of single value {vc.index[0]}')
                if vc.iloc[0]/len(self.train_data)>0.98:
                    config['iskeep'] = False
                    logging.info(f'{feature} should dropp')
                    input()
        write_jsonl(self.feature_json,self.feature_json_path)

    def drop_features(self):
        '''
        根据feature_tabel中的keep列，删除train_data中的特征
        '''
        logging.info(f'==========drop extream features==========')
        drop_features = []
        for feature,config in self.features_config.items():
            if config['iskeep']==False:
                drop_features.append(feature)
        logging.info(f'drop {len(drop_features)} features: {drop_features}')
        self.train_data.drop(columns=drop_features, inplace=True)
        write_jsonl(self.feature_json,self.feature_json_path)

    def reencode(self):
        '''
        based on:
            - self.train_data: 做encoding，确保对于每一个非缺失值，都有对应的encoding_mapping
            - self.feature_json

        side_effect:
            - self.feature_json['label_encoding']['unique_values']
            - self.feature_json['label_encoding']['encoding_mapping']
            - self.feature_json['value_num']
        '''
        for feature, config in self.features_config.items():
            # 跳过没有在train_data中的特征,identifier,非categorical
            # 获取唯一值（包括nan，nan转为字符串'nan'）
            if config['iskeep']==False:
                continue
                
            # 跳过诊断特征，它们将在recode_train_data中特殊处理
            if feature in ['diag_1', 'diag_2', 'diag_3']:
                logging.info(f"跳过诊断特征 {feature} 的重新编码，将在后续特殊处理")
                continue
                
            values = self.train_data[feature]
            unique_values = []
            for v in values.unique():
                if pd.isna(v):
                    logging.info(f'{feature} has nan: {v}')
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
        logging.info(f'==========transfer_all_nan==========')
        nan_values = self.feature_json['nan_values']
        final_nan = None
        replaced_features = []
        id_features = self.ids_mapping.keys()
        for feature,config in tqdm(self.features_config.items(),total=len(self.features_config),desc=f'Processing features'):
            if config['category']=='identifier' or config['type']!='categorical' or config['iskeep']==False:
                continue
            config['missing_values_num'] = 0
            config['missing_values_p'] = 0
            config['missing_values'] = []
            
            # 使用布尔索引一次性替换所有匹配的值
            if feature in id_features:
                # 对于ID特征，先转换为文本进行检查，然后替换缺失值，最后转换回ID
                temp_values = self.train_data[feature].astype(str).map(self.ids_mapping[feature])
                mask = temp_values.isin(nan_values)
                
                # 收集所有的缺失值类型（使用原始ID值）
                missing_vals = set(self.train_data.loc[mask, feature].unique())
                for val in missing_vals:
                    try:
                        val = str(val).strip()
                        if val not in config['missing_values']:
                            config['missing_values'].append(val)
                            if val in config['label_encoding']['unique_values']:
                                config['label_encoding']['unique_values'].remove(val)
                                del config['label_encoding']['encoding_mapping'][val]
                    except:
                        raise ValueError(f"Feature '{feature}' has invalid value: {val}")
                
                # 替换缺失值，保持为ID格式
                self.train_data[feature] = self.train_data[feature].astype(object)
                self.train_data.loc[mask, feature] = final_nan
            else:
                mask = self.train_data[feature].isin(nan_values)
                
                # 收集所有的缺失值类型
                missing_vals = set(self.train_data.loc[mask, feature].unique())
                for val in missing_vals:
                    try:
                        val = str(val).strip()
                        if val not in config['missing_values']:
                            config['missing_values'].append(val)
                            if val in config['label_encoding']['unique_values']:
                                config['label_encoding']['unique_values'].remove(val)
                                del config['label_encoding']['encoding_mapping'][val]
                    except:
                        raise ValueError(f"Feature '{feature}' has invalid value: {val}")
                
                # 替换缺失值
                self.train_data.loc[mask, feature] = self.train_data.loc[mask, feature].astype(object)
                self.train_data.loc[mask, feature] = final_nan
            config['missing_values_num'] = int(self.train_data[feature].isna().sum())
            if config['missing_values_num']>0:
                replaced_features.append(feature)
            config['missing_values_p'] = float(config['missing_values_num']/len(self.train_data[feature]))
            config['missing'] = bool(config['missing_values_num']>0)
            config['label_encoding']['unique_values'] = [str(key) for key in config['label_encoding']['encoding_mapping'].keys()]
            config['label_encoding']['encoding_mapping'] = {val:ind for ind,val in enumerate(config['label_encoding']['unique_values'])}
            config['missing_replace'] = final_nan

            try:
                write_jsonl(self.feature_json, self.feature_json_path)
            except Exception as e:
                print(f"Error writing feature.json: {e}")
                pdb.set_trace()
                print(f'{feature} has invalid value')
                
        if len(replaced_features)>0:
            logging.info(f'replaced {len(replaced_features)} features: {replaced_features}')
        else:
            logging.info('no features replaced')
        logging.info(f'all nan(except id) transferred to {final_nan} in {self.output_dir}/train.csv')

    
    def count_nan(self):
        logging.info(f'==========count_nan==========')
        for feature,config in self.features_config.items():
            if config['category']=='identifier' or config['type']!='categorical' or config['iskeep']==False:
                continue
            nan_num = self.train_data[feature].isna().sum()
            assert config['missing_values_num']==nan_num
            if nan_num>0:
                logging.info(f'{feature}: {nan_num}({nan_num/len(self.train_data)*100:.2f}%)')

    def recode_train_data(self):
        """将所有保留的分类特征根据 label_encoding 映射为整数编码，并保存 CSV。"""

        for feature, config in self.features_config.items():
            if config.get('category') == 'identifier' or config.get('type') != 'categorical' or config.get('iskeep') is False:
                continue

            # 特殊处理诊断特征：使用categorize_diagnosis进行分组
            if feature in ['diag_1', 'diag_2', 'diag_3']:
                logging.info(f"使用categorize_diagnosis处理诊断特征: {feature}")
                self._categorize_diagnosis_feature(feature)
                continue

            # p self.train_data[feature].value_counts()
            # 其他特征使用原有的标签编码方式
            enc_map = config['label_encoding']['encoding_mapping']
            # 先统一转成 str 便于匹配
            self.train_data[feature] = self.train_data[feature].astype(str).map(enc_map)
            # 未映射成功的视为缺失，不填充
            # self.train_data[feature] = self.train_data[feature].fillna(-1).astype('Int64')

        # 保存结果
        self.train_data.to_csv(self.output_dir/'recoded_train.csv', index=False)
        print(f'recoded_train.csv saved to {self.output_dir}/recoded_train.csv')

    def _categorize_diagnosis_feature(self, feature_name):
        """
        对单个诊断特征使用categorize_diagnosis进行分组处理
        
        Args:
            feature_name: 诊断特征名称 (diag_1, diag_2, diag_3)
        """
        # 创建新的分类列名
        new_feature_name = f'new_{feature_name}'
        
        # 复制原始数据
        self.train_data[new_feature_name] = self.train_data[feature_name]
        
        # 保持NaN值为None，不填充为-1
        # self.train_data[new_feature_name] = self.train_data[new_feature_name].fillna(-1)
        
        # 处理包含V或E的编码（外部原因和补充因素）
        # 只处理非NaN的值
        non_null_mask = self.train_data[new_feature_name].notna()
        v_mask = self.train_data.loc[non_null_mask, new_feature_name].astype(str).str.contains('V', na=False)
        e_mask = self.train_data.loc[non_null_mask, new_feature_name].astype(str).str.contains('E', na=False)
        
        self.train_data.loc[non_null_mask & v_mask, new_feature_name] = 0
        self.train_data.loc[non_null_mask & e_mask, new_feature_name] = 0
        
        # 转换为float类型，保持NaN值
        self.train_data[new_feature_name] = pd.to_numeric(self.train_data[new_feature_name], errors='coerce')
        
        # 根据ICD-9编码范围进行分类
        for index, row in self.train_data.iterrows():
            code = row[new_feature_name]
            
            # 跳过NaN值
            if pd.isna(code):
                continue
                
            # 类别1: 循环系统疾病 (390-459, 785)
            if (code >= 390 and code < 460) or (np.floor(code) == 785):
                self.train_data.loc[index, new_feature_name] = 1
            # 类别2: 呼吸系统疾病 (460-519, 786)
            elif (code >= 460 and code < 520) or (np.floor(code) == 786):
                self.train_data.loc[index, new_feature_name] = 2
            # 类别3: 消化系统疾病 (520-579, 787)
            elif (code >= 520 and code < 580) or (np.floor(code) == 787):
                self.train_data.loc[index, new_feature_name] = 3
            # 类别4: 糖尿病 (250)
            elif np.floor(code) == 250:
                self.train_data.loc[index, new_feature_name] = 4
            # 类别5: 损伤和中毒 (800-999)
            elif code >= 800 and code < 1000:
                self.train_data.loc[index, new_feature_name] = 5
            # 类别6: 肌肉骨骼系统疾病 (710-739)
            elif code >= 710 and code < 740:
                self.train_data.loc[index, new_feature_name] = 6
            # 类别7: 泌尿生殖系统疾病 (580-629, 788)
            elif (code >= 580 and code < 630) or (np.floor(code) == 788):
                self.train_data.loc[index, new_feature_name] = 7
            # 类别8: 肿瘤 (140-239)
            elif code >= 140 and code < 240:
                self.train_data.loc[index, new_feature_name] = 8
            # 类别0: 其他疾病
            elif code > 0:
                self.train_data.loc[index, new_feature_name] = 0
        
        # 删除原始诊断列，保留分类后的列
        self.train_data.drop(columns=[feature_name], inplace=True)
        
        # 重命名新列为原始列名
        self.train_data.rename(columns={new_feature_name: feature_name}, inplace=True)
        
        # 更新feature配置
        config = self.features_config[feature_name]
        config['label_encoding']['unique_values'] = ['0', '1', '2', '3', '4', '5', '6', '7', '8']
        config['label_encoding']['encoding_mapping'] = {
            '0': 0,   # 其他疾病
            '1': 1,   # 循环系统疾病
            '2': 2,   # 呼吸系统疾病
            '3': 3,   # 消化系统疾病
            '4': 4,   # 糖尿病
            '5': 5,   # 损伤和中毒
            '6': 6,   # 肌肉骨骼系统疾病
            '7': 7,   # 泌尿生殖系统疾病
            '8': 8    # 肿瘤
        }
        config['value_num'] = 9  # 不包括缺失值
        config['process'] = 'diagnosis_categorized'
        config['missing'] = True  # 标记为有缺失值
        
        logging.info(f"诊断特征 {feature_name} 已分类为9个疾病类别，缺失值保持为None")

    def check_recoded_data(self):
        logging.info('==========check recoded_train.csv==========')
        check_flag = True
        data = pd.read_csv(self.output_dir/'recoded_train.csv')
        error_features = []
        for feature,config in self.features_config.items():
            if config['category']=='identifier' or config['type']!='categorical' or config['iskeep']==False:
                continue
            # print(f'{feature}: {len([int(v) for v in sorted(data[feature].unique()) if pd.notna(v)])}')
            vc = data[feature].value_counts()
            missing_num = config.get('missing_values_num',0)
            missing_count = data[feature].isna().sum()
            if missing_num!=missing_count:
                print(f'{feature}: value_num: {len(vc)}, data_num: {len(data[feature])}')
                print(vc)
                print(f'missing_num: {missing_num} / {missing_count}(counted)')
                print(f'Warning: {feature} has {missing_num} missing values, but {missing_count} counted')
                print('--------------------------------')
                check_flag = False
                error_features.append(feature)
        if check_flag:
            logging.info('recoded_train.csv check passed')
            logging.info(f'>>>>>>>>>>>after recode, remained {len(self.train_data)} rows, {len(self.train_data.columns)} features')
        else:
            logging.error('recoded_train.csv check failed')
            logging.error(f'error_features: {error_features}')

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
        logging.info(f'==========visualize_recoded_features==========')
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
                
            # print(f"处理特征: {feature_name}")
            
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
                # print(f"归一化后范围: [{normalized_min}, {normalized_max}]")
                if not (abs(normalized_max - 1.0) < 1e-6 and abs(normalized_min) < 1e-6):
                    print(f"警告: 特征 {feature_name} 归一化可能不成功")
                    input()
            else:
                print(f"警告: 特征 {feature_name} 的所有值相同 ({feature_max})，设置为0")
                data[feature_name] = 0
                input()
                
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
    # p dp.train_data['admission_type_id'].value_counts()
    dp.config_features()
    dp.clean_invalid_data()
    dp.mark_extream_features()
    dp.drop_features()
    dp.reencode()
    dp.transfer_all_nan()
    # dp.show_feature_config()
    dp.save_train_data('missing_replaced_train')
    dp.count_nan()
    dp.recode_train_data()
    dp.save_train_data('recoded_train')
    dp.check_recoded_data()
    # dp.visualize_recoded_features()
    dp.transfer_to_feature_tabel()

if __name__ == '__main__':
    main()