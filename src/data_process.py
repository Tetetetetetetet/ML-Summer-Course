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
        self.id_mapping = read_jsonl('config/id_mapping.json')

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
        feature_tabel.index.name = 'feature_name'
        feature_tabel.to_csv(self.output_dir/'FeatureTabel.csv', index=True)
        print(f'feature.json transferred to {self.output_dir}/FeatureTabel.csv')

    def count_encounter_repeat(self):
        pass
    def rebuild_unique_values(self):
        '''
        将unique_values按照label_encoding中的encoding_mapping重新排序，并计算unique_values的种类数，记录在feature.json中
        '''
        for feature,config in self.features_config.items():
            if 'label_encoding' in config:
                config['label_encoding']['unique_values'] = sorted(config['label_encoding']['unique_values'])
                config['label_encoding']['encoding_mapping'] = {str(k): v for k, v in config['label_encoding']['encoding_mapping'].items()}

    def transfer_all_nan(self):
        '''
        将所有缺失/无意义值转化为缺失值，并在feature.json中记录缺失值种类
        '''
        nan_values = ['?','None','nan','Unknown/Invalid','']

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
    dp.show_feature_config()
    dp.transfer_to_feature_tabel()

if __name__ == '__main__':
    main()