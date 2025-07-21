import pandas as pd
import json

feature_json = json.load(open('config/feature.json', 'r'))
feature_config = feature_json['features']
all_unique_value_num = 0
feature2num = {}
for feature_name, feature_config in feature_config.items():
    if feature_config['type'] == 'categorical' and feature_config['iskeep']:
        all_unique_value_num += len(feature_config['label_encoding']['unique_values'])
        feature2num[feature_name] = len(feature_config['label_encoding']['unique_values'])
print(all_unique_value_num)
print(feature2num)






