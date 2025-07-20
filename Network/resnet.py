import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# ---------- 1. 数据 ----------
train_df = pd.read_csv('./Data/train.csv')
test_df  = pd.read_csv('./Data/test.csv')

X_train = train_df.iloc[:, :28].astype('float32').values
y_train = train_df['readmitted'].astype('int32').values

X_test  = test_df.iloc[:, :28].astype('float32').values
y_test  = test_df['readmitted'].astype('int32').values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

class_weights = dict(enumerate(len(y_train) / (3 * np.bincount(y_train))))

# ---------- 2. 网络 ----------
def res_block(x, units, dropout_rate):
    shortcut = x
    x = tf.keras.layers.Dense(units, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(units)(x)
    x = tf.keras.layers.Add()([shortcut, x])
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return x

inputs  = tf.keras.layers.Input(shape=(28,))
x = tf.keras.layers.Dense(256, activation='relu')(inputs)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.4)(x)

for _ in range(3):
    x = res_block(x, 256, 0.4)

x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(3, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ---------- 3. 回调 ----------
EPOCHS     = 200
BATCH_SIZE = 64

# ---------- 4. 训练 ----------
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    class_weight=class_weights,
    verbose=2
)

# ---------- 5. 保存 ----------
os.makedirs('./model', exist_ok=True)
save_name = f"./model/{len(history.epoch)}_{BATCH_SIZE}_resnet.h5"
model.save(save_name)
print(f"模型已保存: {save_name}")

# ---------- 6. 评估 ----------
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc:.4f}")

y_pred = model.predict(X_test).argmax(axis=1)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))