from myutils import read_jsonl
import pandas as pd
import json

train_data = pd.read_csv('Dataset/processed/train_processed/logistic_imputed/logistic_imputed_train_try_1.csv')
test_data = pd.read_csv('Dataset/processed/train_processed/logistic_imputed/logistic_imputed_test_try_1.csv')
split_features = ['discharge_disposition_id','medical_specialty']
train_without_split_features = train_data.drop(columns=split_features)
test_without_split_features = test_data.drop(columns=split_features)
train_without_split_features.to_csv('Dataset/processed/train_processed/logistic_imputed/logistic_imputed_train_try_1_without_split_features.csv',index=False)
test_without_split_features.to_csv('Dataset/processed/train_processed/logistic_imputed/logistic_imputed_test_try_1_without_split_features.csv',index=False)
train_split_features = train_data[split_features+['readmitted']]
test_split_features = test_data[split_features+['readmitted']]
train_split_features.to_csv('Dataset/processed/train_processed/logistic_imputed/logistic_imputed_train_try_1_split_features.csv',index=False)
test_split_features.to_csv('Dataset/processed/train_processed/logistic_imputed/logistic_imputed_test_try_1_split_features.csv',index=False)








