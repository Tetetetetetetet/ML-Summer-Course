from myutils import read_jsonl,write_jsonl

def main():
    feature_json = read_jsonl('Dataset/feature.json')
    features = feature_json['features']
    for feature,info in features.items():
        if feature not in ['max_glu_serum','weight']:
            features[feature]['iskeep'] = True
        else:
            features[feature]['iskeep'] = False
    write_jsonl('feature.json', feature_json)
if __name__ == '__main__':
    main()