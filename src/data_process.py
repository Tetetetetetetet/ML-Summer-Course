from myutils import read_jsonl,write_jsonl

class DataProcess:
    def __init__(self):
        self.feature_json_path = 'feature.json'
        self.feature_json = read_jsonl(self.feature_json_path)
        self.features_config = self.feature_json['features']

    def split_features(self):
        self.features_config = self.feature_json['features']
        self.features_config = {k:v for k,v in self.features_config.items() if v['iskeep']}
        self.features_config = list(self.features_config.keys())
        self.features_config = {k:v for k,v in self.features_config.items() if v['iskeep']}

    def config_features(self):
        for feature_name,feature_info in self.features_config.items():
            if feature_name in ['max_glu_serum','weight']:
                feature_info['iskeep'] = False
            else:
                feature_info['iskeep'] = True
            if feature_info['type'] == 'categorical':
                pass
        write_jsonl(self.features_config,self.feature_json_path)

    def process(self):
        for feature,info in self.features.items():

def main():
    feature_json = read_jsonl('feature.json')
    features = feature_json['features']

if __name__ == '__main__':
    main()