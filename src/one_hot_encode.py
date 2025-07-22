import pandas as pd
import json
from sklearn.preprocessing import OneHotEncoder

# 文件路径
train_path = '../Dataset/processed/train_processed/improved_logistic_imputed/improved_logistic_imputed_train_final.csv'
test_path = '../Dataset/processed/train_processed/improved_logistic_imputed/improved_logistic_imputed_test_final.csv'
feature_json_path = '../config/feature.json'
train_output_path = '../Dataset/processed/train_processed/improved_logistic_imputed/one_hot_improved_logistic_imputed_train_final.csv'
test_output_path = '../Dataset/processed/train_processed/improved_logistic_imputed/one_hot_improved_logistic_imputed_test_final.csv'

# 加载数据
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# 加载特征信息
with open(feature_json_path, 'r') as f:
    feature_info = json.load(f)['features']

# 决定需要 One-Hot 编码的特征
onehot_features = []
excluded_features = []

for name, info in feature_info.items():
    if info['type'] == 'categorical' and name != 'age' and name != 'readmitted':
        if info.get('value_num', 0) <= 30:
            onehot_features.append(name)
        else:
            excluded_features.append(name)

# 提取需要编码的列
X_train_cat = train_df[onehot_features]
X_test_cat = test_df[onehot_features]

# 创建 OneHotEncoder
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoder.fit(X_train_cat)  # 使用训练集确定所有可能的类别组合

# 转换
X_train_encoded = encoder.transform(X_train_cat)
X_test_encoded = encoder.transform(X_test_cat)

# 获取编码后的列名
encoded_col_names = encoder.get_feature_names_out(onehot_features)

# 创建编码后的 DataFrame，并转为 int 类型（0/1）
X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=encoded_col_names, index=train_df.index).astype(int)
X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=encoded_col_names, index=test_df.index).astype(int)

# 将原始数据中被编码的列删掉
train_df_clean = train_df.drop(columns=onehot_features)
test_df_clean = test_df.drop(columns=onehot_features)

# 合并原始数值列和编码列
train_final = pd.concat([train_df_clean, X_train_encoded_df], axis=1)
test_final = pd.concat([test_df_clean, X_test_encoded_df], axis=1)

# 保存到 CSV 文件
train_final.to_csv(train_output_path, index=False)
test_final.to_csv(test_output_path, index=False)

# 结果展示
print("✅ 编码完成，并已保存为 0/1 整数值！")
print(f"训练集输出文件：{train_output_path}")
print(f"测试集输出文件：{test_output_path}")
print(f"One-Hot 编码特征数：{len(encoded_col_names)}")
print("被编码的特征包括：")
for feat in onehot_features:
    print(f"- {feat}")
print("未编码的（因类别数 > 30 或被排除）：")
for feat in excluded_features:
    print(f"- {feat}")


"""
✅ 编码完成，并已保存为 0/1 整数值！
训练集输出文件：../Dataset/processed/train_processed/improved_logistic_imputed/one_hot_improved_logistic_imputed_train_final.csv
测试集输出文件：../Dataset/processed/train_processed/improved_logistic_imputed/one_hot_improved_logistic_imputed_test_final.csv
One-Hot 编码特征数：178
被编码的特征包括：
- race
- gender
- weight
- admission_type_id
- discharge_disposition_id
- admission_source_id
- payer_code
- diag_1
- diag_2
- diag_3
- max_glu_serum
- A1Cresult
- metformin
- repaglinide
- nateglinide
- chlorpropamide
- glimepiride
- acetohexamide
- glipizide
- glyburide
- tolbutamide
- pioglitazone
- rosiglitazone
- acarbose
- miglitol
- troglitazone
- tolazamide
- examide
- citoglipton
- insulin
- glyburide-metformin
- glipizide-metformin
- glimepiride-pioglitazone
- metformin-rosiglitazone
- metformin-pioglitazone
- change
- diabetesMed
未编码的（因类别数 > 30 或被排除）：
- medical_specialty

[INFO] 实验名称: normal_all_LogisticRegression_None_oversample
[INFO] 数据集模式: normal
[INFO] 是否过采样: True
[INFO] 最佳模型: LogisticRegression_tuned
[INFO] 最佳准确率: 0.4916
[INFO] 最佳F1-Macro: 0.4318
[INFO] 最佳AUC: 0.6533
[INFO] ==========建模流程完成==========

[INFO] 实验名称: normal_all_GradientBoosting_None_oversample
[INFO] 数据集模式: normal
[INFO] 是否过采样: True
[INFO] 最佳模型: GradientBoosting
[INFO] 最佳准确率: 0.5703
[INFO] 最佳F1-Macro: 0.4247
[INFO] 最佳AUC: 0.6742
[INFO] ==========建模流程完成==========

[INFO] ==========建模总结==========
[INFO] 实验名称: normal_all_RandomForest_None_oversample
[INFO] 数据集模式: normal
[INFO] 是否过采样: True
[INFO] 最佳模型: RandomForest
[INFO] 最佳准确率: 0.5821
[INFO] 最佳F1-Macro: 0.4189
[INFO] 最佳AUC: 0.6579
[INFO] ==========建模流程完成==========
"""
