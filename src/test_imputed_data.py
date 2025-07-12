from myutils import *
import pdb
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    train_data = pd.read_csv('Dataset/processed/train_processed/logistic_imputed/logistic_imputed_train_final.csv')
    test_data = pd.read_csv('Dataset/processed/train_processed/logistic_imputed/logistic_imputed_test_final.csv')
    feature_json = read_jsonl('config/feature.json')
    feature_config = feature_json['features']
    error_features = []
    for feature, config in feature_config.items():
        if config['iskeep'] == False or config['type'] != 'categorical':
            continue
        train_vc = train_data[feature].value_counts()
        test_vc = test_data[feature].value_counts()
        logging.info(f'train value counts: {train_vc}\nnan num: {train_data[feature].isna().sum()}\nvalue range: {train_data[feature].min()} - {train_data[feature].max()}')
        logging.info(f'test value counts: {test_vc}\nnan num: {test_data[feature].isna().sum()}\nvalue range: {test_data[feature].min()} - {test_data[feature].max()}')
        pdb.set_trace()


if __name__ == "__main__":
    main()