import pandas as pd
from myutils import read_jsonl

def main():
    train_data = pd.read_csv('Dataset/processed/train_processed/improved_logistic_imputed/improved_logistic_imputed_train_final.csv')
    test_data = pd.read_csv('Dataset/processed/train_processed/improved_logistic_imputed/improved_logistic_imputed_test_final.csv')
    feature_json = read_jsonl('config/feature.json')
    feature_config = feature_json['features']
    for feature,config in feature_config.items():
        if config['iskeep'] == False:
            train_data = train_data.drop(columns=[feature])
            test_data = test_data.drop(columns=[feature])
    train_data.to_csv('Dataset/processed/train_processed/improved_logistic_imputed/improved_logistic_imputed_train_final_selected.csv',index=False)
    test_data.to_csv('Dataset/processed/train_processed/improved_logistic_imputed/improved_logistic_imputed_test_final_selected.csv',index=False)


if __name__ == '__main__':
    main()