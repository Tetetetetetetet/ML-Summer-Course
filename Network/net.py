import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix


# -------------------------------------------------
# 1. 构建模型
# -------------------------------------------------
def build_model(feature_cols=None, n=None):
    """
    构建并返回模型。
    参数：
        feature_cols : 指定要使用的特征列编号（1-based 列表或 range）。
                       若为 None，则默认使用前 n 个特征（1..n）。
        n            : 当 feature_cols=None 时生效；使用前 n 个特征。
    返回：
        model, actual_feature_cols
    """
    if feature_cols is None:
        if n is None:
            raise ValueError("必须指定 feature_cols 或 n")
        feature_cols = list(range(1, n + 1))
    else:
        feature_cols = list(feature_cols)

    input_dim = len(feature_cols)

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

    inputs = tf.keras.layers.Input(shape=(input_dim,))
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
    return model, feature_cols


# -------------------------------------------------
# 2. 训练
# -------------------------------------------------
def train(train_csv: str,
          feature_cols=None,
          n=None,
          epochs: int = 200,
          batch_size: int = 64,
          save_path: str = './model/best_resnet.h5'):
    """
    读取训练集 -> 训练 -> 保存模型
    参数：
        train_csv    : 训练集 csv 路径
        feature_cols : 同 build_model
        n            : 同 build_model
        epochs       : 训练轮数
        batch_size   : 批次大小
        save_path    : 模型保存路径（含文件名）
    """
    # 1) 读数据
    train_df = pd.read_csv(train_csv)
    model, cols = build_model(feature_cols=feature_cols, n=n)

    # 取对应列（pandas 列号从 0 开始，csv 第 1 列是 0 号）
    col_idx = [c - 1 for c in cols]  # 转为 0-based
    X = train_df.iloc[:, col_idx].astype('float32').values
    y = train_df['readmitted'].astype('int32').values

    # 2) 标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 3) 类别权重
    class_weights = dict(enumerate(len(y) / (3 * np.bincount(y))))

    # 4) 训练
    model.fit(X, y,
              epochs=epochs,
              batch_size=batch_size,
              validation_split=0.2,
              class_weight=class_weights,
              verbose=2)

    # 5) 保存
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f'模型已保存: {save_path}')


# -------------------------------------------------
# 3. 评估
# -------------------------------------------------
def evaluate(test_csv: str,
             model_path: str,
             feature_cols=None,
             n=None):
    """
    读取测试集 -> 加载模型 -> 打印评估结果
    参数：
        test_csv     : 测试集 csv 路径
        model_path   : 已保存的模型文件路径
        feature_cols : 同 build_model
        n            : 同 build_model
    """
    # 1) 读数据
    test_df = pd.read_csv(test_csv)
    _, cols = build_model(feature_cols=feature_cols, n=n)
    col_idx = [c - 1 for c in cols]
    X = test_df.iloc[:, col_idx].astype('float32').values
    y = test_df['readmitted'].astype('int32').values

    # 2) 标准化（复用训练集统计量更严谨，这里简单演示）
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 3) 加载模型
    model = tf.keras.models.load_model(model_path)

    # 4) 评估
    loss, acc = model.evaluate(X, y, verbose=0)
    print(f'Test Accuracy: {acc:.4f}')

    y_pred = model.predict(X).argmax(axis=1)
    print("Classification Report:\n", classification_report(y, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y, y_pred))


# -------------------------------------------------
# 4. 命令行示例（可选）
# -------------------------------------------------
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train/Evaluate ResNet-like model.')
    subparsers = parser.add_subparsers(dest='command')

    # train
    p_train = subparsers.add_parser('train')
    p_train.add_argument('--train_csv', default='./Data/train.csv')
    p_train.add_argument('--feature_cols', type=int, nargs='*', default=None)
    p_train.add_argument('--n', type=int, default=None)
    p_train.add_argument('--epochs', type=int, default=200)
    p_train.add_argument('--batch_size', type=int, default=64)
    p_train.add_argument('--save_path', default='./model/best_resnet.h5')

    # evaluate
    p_eval = subparsers.add_parser('eval')
    p_eval.add_argument('--test_csv', default='./Data/test.csv')
    p_eval.add_argument('--model_path', required=True)
    p_eval.add_argument('--feature_cols', type=int, nargs='*', default=None)
    p_eval.add_argument('--n', type=int, default=None)

    args = parser.parse_args()

    if args.command == 'train':
        train(args.train_csv,
              feature_cols=args.feature_cols,
              n=args.n,
              epochs=args.epochs,
              batch_size=args.batch_size,
              save_path=args.save_path)
    elif args.command == 'eval':
        evaluate(args.test_csv,
                 args.model_path,
                 feature_cols=args.feature_cols,
                 n=args.n)
    else:
        parser.print_help()