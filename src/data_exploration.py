import pandas as pd
import tqdm
import numpy as np
from matplotlib import pyplot as plt
from myutils import *
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s\n',filename='config/data_exploration.md',filemode='w')

train_data = pd.read_csv('Dataset/processed/train_processed/recoded_train.csv')
test_data = pd.read_csv('Dataset/processed/test_processed/recoded_test.csv')
train_data.head()
feature_json = read_jsonl('config/feature.json')
feature_config = feature_json['features']
features = train_data.columns.tolist()
logging.info(f'{len(features)} features: {features}')

class DataExploration:
    def __init__(self,train_data:pd.DataFrame,test_data:pd.DataFrame,feature_config:dict,target:str):
        self.train_data = train_data
        self.test_data = test_data
        self.feature_config = feature_config
        self.features = train_data.columns.tolist()
        self.target = target
        
    def show_category_var_relation(self,feature:str,data:pd.DataFrame):
        target = self.target
        data = data.fillna(-1)
        vc = data[feature].value_counts()
        logging.info(f"{vc}")
        mapping = feature_config[feature]['label_encoding']['encoding_mapping']
        id2name = {v: k for k, v in mapping.items()}
        logging.info(f"{id2name}")
        statics = {'class':[],'class_name':[],'mean':[],'std':[],'min':[],'max':[],'median':[],'mode':[],'count':[]}
        logging.info("### statics")
        for name,group in data.groupby(feature):
            statics['class'].append(name)
            statics['class_name'].append(id2name[name] if name in id2name else 'N/A')
            statics['mean'].append(group[target].mean())
            statics['std'].append(group[target].std())
            statics['min'].append(group[target].min())
            statics['max'].append(group[target].max())
            statics['median'].append(group[target].median())
            statics['mode'].append(group[target].mode().iloc[0] if len(group[target].mode()) > 0 else None)
            statics['count'].append(group[target].count())
        df = pd.DataFrame(statics)
        logging.info(f"{df.to_markdown()}")
        # 计算相关系数
        corr = data[[feature, target]].corr()
        logging.info(f"{feature} 与 {target} 的相关系数矩阵：")
        logging.info(corr.to_markdown())
        logging.info(f"{feature} 与 {target} 的皮尔逊相关系数为: {corr.loc[feature, target]}")

    def show_numerical_var_relation(self,feature:str,data:pd.DataFrame):
        target = self.target
        data = data.fillna(-1)
        corr = data[[feature, target]].corr()
        logging.info(f"{feature} 与 {target} 的皮尔逊相关系数为: {corr.loc[feature, target]}")

de = DataExploration(train_data,test_data,feature_config,'readmitted')
for i,feature in tqdm.tqdm(enumerate(features),desc='exploring features'):
    config = feature_config[feature]
    if config['category'] == 'numerical' or config['category'] == 'identifier':
        continue
    logging.info(f"## {feature}: ({config['category']}, {config['type']})")
    if config['category'] == 'categorical':
        de.show_category_var_relation(feature,train_data)
    else:
        de.show_numerical_var_relation(feature,train_data)