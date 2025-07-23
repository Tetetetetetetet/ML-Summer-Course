import pandas as pd
import tqdm
import json

# resnet_data_train = pd.read_csv('Dataset/processed/train_processed/network_train.csv')
# resnet_data_test = pd.read_csv('Dataset/processed/train_processed/network_test.csv')
resnet_data_train = pd.read_csv('Dataset/processed/train_processed/logistic_imputed/logistic_imputed_train_final.csv')
resnet_data_test = pd.read_csv('Dataset/processed/train_processed/logistic_imputed/logistic_imputed_test_final.csv')
# original_data_train = pd.read_csv('Dataset/processed/train_processed/logistic_imputed/logistic_imputed_train_try.csv')
# original_data_test = pd.read_csv('Dataset/processed/train_processed/logistic_imputed/logistic_imputed_test_try.csv')
original_data_train = pd.read_csv('Dataset/processed/train_processed/logistic_imputed/logistic_imputed_train_try.csv')
original_data_test = pd.read_csv('Dataset/processed/train_processed/logistic_imputed/logistic_imputed_test_try.csv')
missing_features = []
feature_json = json.load(open('config/feature.json', 'r'))
feature_config = feature_json['features']
"""
for feature_name, feature_config in feature_config.items():
    if feature_config.get('missing', False) and feature_name in resnet_data_test.columns:
        resnet_data_train.drop(feature_name, axis=1, inplace=True)
        resnet_data_test.drop(feature_name, axis=1, inplace=True)
        original_data_train.drop(feature_name, axis=1, inplace=True)
        original_data_test.drop(feature_name, axis=1, inplace=True)
        missing_features.append(feature_name)
"""
print(f'num of missing_features: {len(missing_features)}')
print(resnet_data_train.shape)
print(original_data_train.shape)
print(resnet_data_test.shape)
print(original_data_test.shape)
# 将特征顺序调整后对比两个数据集
original_data_train = original_data_train.loc[:, sorted(resnet_data_train.columns)]
original_data_test = original_data_test.loc[:, sorted(resnet_data_test.columns)]
resnet_data_train = resnet_data_train.loc[:, sorted(resnet_data_train.columns)]
resnet_data_test = resnet_data_test.loc[:, sorted(resnet_data_test.columns)]
# 说明是否全为true
print((original_data_train.columns==resnet_data_train.columns).all())
print((original_data_test.columns==resnet_data_test.columns).all())
# 找出存在值不相同的列，打印其列名
print("=== 检查列值差异 ===")
different_columns = []
for col in original_data_train.columns:
    if not (original_data_train[col].values == resnet_data_train[col].values).all():
        different_columns.append(col)
        print(f"列 '{col}' 存在差异")
        # 显示差异的统计信息
        diff_count = (original_data_train[col].values != resnet_data_train[col].values).sum()
        print(f"  差异数量: {diff_count}/{len(original_data_train)} ({diff_count/len(original_data_train)*100:.2f}%)")

if not different_columns:
    print("所有列的值都相同")
# print(sum(original_data_train.values!=resnet_data_train.values))
# print(sum(original_data_test.values!=resnet_data_test.values))
# print(resnet_data_train.isna().sum())
# print(original_data_train.isna().sum())
# print(resnet_data_test.isna().sum())
# print(original_data_test.isna().sum())

# 测试集检验
print("\n测试集检验:")
missing_in_test = 0
for idx, row in tqdm.tqdm(list(resnet_data_test.iterrows())):
    # 检查这一行是否在original_data_test中存在
    mask = (original_data_test == row).all(axis=1)
    if not mask.any():
        missing_in_test += 1
        if missing_in_test <= 5:  # 只打印前5个缺失样本的信息
            print(f"  缺失样本 {idx}: {row.iloc[:5].tolist()}...")  # 只显示前5个值

print(f"测试集中缺失的样本数量: {missing_in_test}/{len(resnet_data_test)} ({missing_in_test/len(resnet_data_test)*100:.2f}%)")

# 反向检验：原始数据集中有多少样本在Network数据集中
print("\n=== 反向检验 ===")
print("原始训练集检验:")
missing_in_original_train = 0
for idx, row in original_data_train.iterrows():
    mask = (resnet_data_train == row).all(axis=1)
    if not mask.any():
        missing_in_original_train += 1

print(f"原始训练集中缺失的样本数量: {missing_in_original_train}/{len(original_data_train)} ({missing_in_original_train/len(original_data_train)*100:.2f}%)")

print("原始测试集检验:")
missing_in_original_test = 0
for idx, row in original_data_test.iterrows():
    mask = (resnet_data_test == row).all(axis=1)
    if not mask.any():
        missing_in_original_test += 1

print(f"原始测试集中缺失的样本数量: {missing_in_original_test}/{len(original_data_test)} ({missing_in_original_test/len(original_data_test)*100:.2f}%)")
