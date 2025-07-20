import pandas as pd

resnet_data_train = pd.read_csv('Network/Data/train.csv')
resnet_data_test = pd.read_csv('Network/Data/test.csv')
original_data_train = pd.read_csv('Dataset/processed/train_processed/improved_logistic_imputed/improved_logistic_imputed_train_final_selected.csv')
original_data_test = pd.read_csv('Dataset/processed/train_processed/improved_logistic_imputed/improved_logistic_imputed_test_final_selected.csv')

print(resnet_data_train.shape)
print(original_data_train.shape)
print(resnet_data_test.shape)
print(original_data_test.shape)
# 将特征顺序调整后对比两个数据集
original_data_train = original_data_train.loc[:, sorted(resnet_data_train.columns)]
original_data_test = original_data_test.loc[:, sorted(resnet_data_test.columns)]
resnet_data_train = resnet_data_train.loc[:, sorted(resnet_data_train.columns)]
resnet_data_test = resnet_data_test.loc[:, sorted(resnet_data_test.columns)]
print(original_data_train.columns==resnet_data_train.columns)
print(original_data_test.columns==resnet_data_test.columns)
print(sum(original_data_train.values!=resnet_data_train.values))
print(sum(original_data_test.values!=resnet_data_test.values))
