from myutils import read_jsonl,write_jsonl
from pathlib import Path
import pdb
import pandas as pd

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

    def reencode(self):
        '''
        根据原始数据，重新做encoding，并要求encoding_mapping和unique_values一一对应
        '''
        for feature, config in self.features_config.items():
            # 跳过没有在train_data中的特征,identifier,非categorical
            # 获取唯一值（包括nan，nan转为字符串'nan'）
            values = self.train_data[feature]
            unique_values = []
            for v in values.unique():
                if pd.isna(v):
                    unique_values.append('nan')
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
            - ismissing: True, False
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
        transfer feature.json to FeatureTabel.csv
        '''
        feature_tabel = pd.DataFrame(self.features_config).T
        feature_tabel.drop(columns=['visualization','label_encoding'], inplace=True)
        feature_tabel.index.name = 'feature_name'
        feature_tabel.to_csv(self.output_dir/'FeatureTabel.csv', index=True)
        print(f'feature.json transferred to {self.output_dir}/FeatureTabel.csv')

    def count_encounter_repeat(self):
        pass

    def transfer_all_nan(self):
        '''
        将所有缺失/无意义值转化为缺失值，并在feature.json中记录缺失值种类
        '''
        nan_values = ['?','None','nan','Unknown/Invalid','NULL','Not Available','Not Mapped']
        final_nan = 'missing'
        for feature,config in self.features_config.items():
            if config['category']=='identifier' or config['type']!='categorical':
                continue
            config['missing_values_num'] = 0
            config['missing_values_p'] = 0
            config['missing_values'] = []
            for idx,data in enumerate(self.train_data[feature]):
                if data in nan_values:
                    config['label_encoding']['unique_values'].remove(data)
                    config['label_encoding']['unique_values'].append(final_nan)
                    config['label_encoding']['encoding_mapping'][final_nan] = config['label_encoding']['encoding_mapping'][data]
                    del config['label_encoding']['encoding_mapping'][data]
                    config['missing_values_num']+=1
                    if data not in config['missing_values']:
                        config['missing_values'].append(data)
                    self.train_data.loc[idx,feature] = final_nan
            config['missing_values_p'] = config['missing_values_num']/len(self.train_data[feature])
            config['label_encoding']['unique_values'] = [key for key in config['label_encoding']['encoding_mapping'].keys()]
        self.train_data.to_csv(self.output_dir/'train.csv', index=False)
        write_jsonl(self.feature_json,self.feature_json_path)
        print(f'all nan(except id) transferred to {final_nan} in {self.output_dir}/train.csv')

    def transfer_all_nan_for_id(self):
        '''
        对于ids_mapping中的三个feature（admission_type_id, discharge_disposition_id, admission_source_id），
        其值本身为id，是否为nan需要通过ids_mapping对应的真实含义判断（如"Unknown/Invalid"、"NULL"、"Not Available"、"Not Mapped"等）。
        将这些无意义的id统一转为'missing'，并在feature.json中记录。
        '''
        # 需要处理的特征
        id_features = ['admission_type_id', 'discharge_disposition_id', 'admission_source_id']
        # 这些id对应的无效含义
        nan_meanings = set(['Unknown/Invalid', 'NULL', 'Not Available', 'Not Mapped', 'nan', '?', 'None'])
        final_nan = 'missing'

        ids_mapping = self.ids_mapping

        for feature in id_features:
            if feature not in self.features_config:
                raise ValueError(f"Feature '{feature}' not found in features_config")
            config = self.features_config[feature]
            # 获取id->含义的映射
            id2meaning = ids_mapping.get(feature, {})
            # 找出所有无效id
            invalid_ids = [id_ for id_, meaning in id2meaning.items() if str(meaning).strip() in nan_meanings]
            # 还要考虑feature.json中label_encoding的unique_values和encoding_mapping
            unique_values = config['label_encoding']['unique_values']
            encoding_mapping = config['label_encoding']['encoding_mapping']
            # 统计
            config['missing_values_num'] = 0
            config['missing_values_p'] = 0
            config['missing_values'] = []
            # 记录哪些id被视为missing
            missing_ids = []
            # 遍历数据，替换无效id为'missing'
            for idx, data in enumerate(self.train_data[feature]):
                # 注意data可能是int/float/str
                data_str = str(int(data)) if pd.notna(data) and str(data).isdigit() else str(data)
                if data_str in [str(i) for i in invalid_ids]:
                    # 替换为'missing'
                    self.train_data.loc[idx, feature] = final_nan
                    config['missing_values_num'] += 1
                    missing_ids.append(data_str)
            # 更新unique_values和encoding_mapping
            # 先移除无效id
            for invalid_id in [str(i) for i in invalid_ids]:
                if invalid_id in unique_values:
                    unique_values.remove(invalid_id)
                if invalid_id in encoding_mapping:
                    # 将其编码赋给'missing'
                    encoding_mapping[final_nan] = encoding_mapping[invalid_id]
                    del encoding_mapping[invalid_id]
            # 如果'missing'不在unique_values，添加
            if final_nan not in unique_values:
                unique_values.append(final_nan)
            # 重新整理unique_values顺序
            label_encoding['unique_values'] = [k for k in encoding_mapping.keys()]
            # 记录missing比例
            config['missing_values_p'] = config['missing_values_num'] / len(self.train_data[feature])
            config['missing_values'] = missing_ids
            # 更新回features_config
            config['label_encoding'] = label_encoding

        # 保存结果
        self.train_data.to_csv(self.output_dir / 'train.csv', index=False)
        write_jsonl(self.feature_json, self.feature_json_path)
        print(f"All nan/invalid ids in {id_features} transferred to '{final_nan}' in {self.output_dir}/train.csv")

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


def main():
    dp = DataProcess()
    dp.reencode()
    dp.config_features()
    dp.transfer_to_feature_tabel()
    dp.transfer_all_nan()
    dp.transfer_all_nan_for_id()
    dp.show_feature_config()

if __name__ == '__main__':
    main()