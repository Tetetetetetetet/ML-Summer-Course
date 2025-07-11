import pandas as pd
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
        self.train_data = pd.read_csv('Dataset/diabetic_data_training.csv')
        self.output_dir = Path('Dataset/processed')
        self.dataset_path = Path('Dataset')
        self.ids_mapping = read_jsonl('config/id_mapping.json')
    
    def load_data(self):
        """
        1. 加载数据
        """
        logging.info("==========load_data==========")
        self.test_data = pd.read_csv('Dataset/diabetic_data_test.csv')

    def drop_features(self):
        """
        删除特征:
        - iskeep==False
        """
        logging.info("==========drop_features==========")
        self.test_data = self.test_data.drop(columns=['encounter_id','patient_nbr'])
        for feature_name,config in self.features_config.items():
            if not config['iskeep']:
                self.test_data = self.test_data.drop(columns=[feature_name])

        logging.info(f"删除特征后，剩余{len(self.test_data.columns)}特征: {self.test_data.columns}")

    def map_test_data(self):
        """
        map data to id
        """
        logging.info("==========map_test_data==========")
        for feature_name,config in self.features_config.items():
            if config['type'] != 'categorical':
                self.test_data[feature_name] = self.test_data[feature_name].map(config['label_encoding']['encoding_mapping'])
        
def main():
    handler = DataProcessTest()
    handler.load_data()
    handler.drop_features()
    handler.map_test_data()
    handler.test_data.to_csv('Dataset/processed/test_data.csv', index=False)

if __name__ == '__main__':
    main()