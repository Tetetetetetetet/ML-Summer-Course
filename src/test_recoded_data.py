from myutils import *
import pandas as pd
import pdb

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    train_data = pd.read_csv('Dataset/processed/train_processed/recoded_train.csv')
    test_data = pd.read_csv('Dataset/processed/test_processed/recoded_test.csv')
    feature_json = read_jsonl('config/feature.json')
    feature_config = feature_json['features']
    error_features = []
    for feature, config in feature_config.items():
        if config['iskeep'] == False or config['type'] != 'categorical':
            continue
        train_vc = train_data[feature].value_counts()
        test_vc = test_data[feature].value_counts()
        logging.info(f'train value counts: {train_vc}\nnan num: {train_data[feature].isna().sum()}')
        if train_vc.shape[0] != test_vc.shape[0]:
            for val in test_data[feature].unique():
                if (not pd.isna(val)) and (val not in train_data[feature].unique()):
                    logging.warning(f'{feature} 在训练集和测试集的取值个数不同, 测试集有 {val}, 训练集没有')
                    pdb.set_trace()
                    error_features.append(feature)
        if train_vc.shape[0] == 1:
            logging.warning(f'{feature} 仅有一个取值')
            error_features.append(feature)
            pdb.set_trace()
        train_nan_num = train_data[feature].isna().sum()
        test_nan_num = test_data[feature].isna().sum()
        try:
            assert train_nan_num == config['missing_values_num']
            assert test_nan_num == config['missing_in_test_num']
        except AssertionError:
            logging.warning(f'{feature} 在训练集和测试集的缺失值个数分别为 {train_nan_num} 和 {test_nan_num}, 应该分别为 {config["missing_values_num"]} 和 {config["missing_in_test_num"]}')
            error_features.append(feature)
            pdb.set_trace()
        # logging.info(f'{feature} 在训练集和测试集的缺失值个数分别为 {train_nan_num} 和 {test_nan_num}')
    if len(error_features) > 0:
        logging.error(f'以下特征在训练集和测试集的取值个数不同: {error_features}')
    else:
        logging.info('所有特征在训练集和测试集的取值个数相同')



if __name__ == "__main__":
    main()