import pandas as pd
import numpy as np
import os
import logging
import pdb

class DataFit:
    def __init__(self):
        self.output_dir = 'Dataset/processed'
        self.train_data = pd.read_csv('Dataset/processed/normalized_train.csv')
        self.test_data = pd.read_csv('Dataset/processed/normalized_test.csv')
        self.feature_json = json.load(open('Dataset/feature.json'))
        self.features_config = self.feature_json['features_config']
        self.features_config_path = 'Dataset/features_config.json'
        self.features_config = json.load(open(self.features_config_path))
        self.features_config_path = 'Dataset/features_config.json'
    def load_data(self,impute_way:str='knn'):
        """
        1. 加载数据
        """
        logging.info("==========load_data==========")
        if impute_way == 'knn':
            self.train_data = pd.read_csv('Dataset/processed/imputed/knn_imputed_train.csv')
        elif impute_way == 'mean':
            self.train_data = pd.read_csv('Dataset/processed/mean_imputed_train.csv')
        elif impute_way == 'median':
            self.train_data = pd.read_csv('Dataset/processed/median_imputed_train.csv')

    def fit_data(self):
        """
        1. 对数据进行拟合
        """
        logging.info("==========fit_data==========")
        self.train_data = self.train_data.drop(columns=['id'])
        self.test_data = self.test_data.drop(columns=['id'])
        self.train_data = self.train_data.drop(columns=['target'])
        self.test_data = self.test_data.drop(columns=['target'])
        self.train_data = self.train_data.drop(columns=['target'])

    def fit_data_with_model(self):