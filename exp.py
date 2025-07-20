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
print(resnet_data_train.isna().sum())
print(original_data_train.isna().sum())
print(resnet_data_test.isna().sum())
print(original_data_test.isna().sum())

# 逐行检验Network数据集的样本是否在原始数据集中存在
print("\n=== 逐行检验样本匹配情况 ===")

# 训练集检验
print("训练集检验:")
missing_in_train = 0
for idx, row in resnet_data_train.iterrows():
    # 检查这一行是否在original_data_train中存在
    # 使用所有列进行匹配
    mask = (original_data_train == row).all(axis=1)
    if not mask.any():
        missing_in_train += 1
        if missing_in_train <= 5:  # 只打印前5个缺失样本的信息
            print(f"  缺失样本 {idx}: {row.iloc[:5].tolist()}...")  # 只显示前5个值

print(f"训练集中缺失的样本数量: {missing_in_train}/{len(resnet_data_train)} ({missing_in_train/len(resnet_data_train)*100:.2f}%)")

# 测试集检验
print("\n测试集检验:")
missing_in_test = 0
for idx, row in resnet_data_test.iterrows():
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
