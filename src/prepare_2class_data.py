import pandas as pd
from myutils import read_jsonl
import pdb
import logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def main():
    train_data = pd.read_csv('Dataset/processed/train_processed/logistic_imputed/logistic_imputed_train_final.csv')
    test_data = pd.read_csv('Dataset/processed/train_processed/logistic_imputed/logistic_imputed_test_final.csv')
    train_data['readmitted'] = train_data['readmitted'].apply(lambda x: 0 if x == 0 else 1)
    test_data['readmitted'] = test_data['readmitted'].apply(lambda x: 0 if x == 0 else 1)
    train_data.to_csv('Dataset/processed/train_processed/logistic_imputed/logistic_imputed_train_final_2class.csv', index=False)
    test_data.to_csv('Dataset/processed/train_processed/logistic_imputed/logistic_imputed_test_final_2class.csv', index=False)
    logging.info(f'train data shape: {train_data.shape}\ntest data shape: {test_data.shape}')

if __name__ == "__main__":
    main()