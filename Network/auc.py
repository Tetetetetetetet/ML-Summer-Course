import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import roc_auc_score

# 1. 读数据
test_df = pd.read_csv('./Data/test.csv')

# 2. 取特征 & 标签
X_test = test_df.iloc[:, :28].astype('float32').values
y_test = test_df['readmitted'].astype('int32').values   # 0,1,2

# 3. 与训练时一致的归一化
# 你训练时用的 StandardScaler，这里要复用同一个 scaler
# 如果训练脚本里保存了 scaler，直接加载；否则只能重新 fit（会略有偏差）
# 下面演示“重新 fit”的近似做法
scaler = StandardScaler()
# 为了对齐，用训练集均值/方差才最准确；
# 这里简单用测试集做示例（实际应加载训练 scaler）
X_test = scaler.fit_transform(X_test)

# 4. 加载模型
model = tf.keras.models.load_model('./model/100_64_resnet.h5')

# 5. 预测概率
y_score = model.predict(X_test)      # shape=(n_samples, 3)

# 6. 计算多类别 ROC-AUC（宏平均）
# 先把标签 one-hot：[[1,0,0], [0,1,0], ...]
y_true_bin = label_binarize(y_test, classes=[0, 1, 2])
auc_macro  = roc_auc_score(y_true_bin, y_score,
                           average='macro', multi_class='ovr')

print(f"宏平均 ROC-AUC = {auc_macro:.4f}")