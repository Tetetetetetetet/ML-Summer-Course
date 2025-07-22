#!/usr/bin/env python3
"""
测试GPU训练功能
"""

import tensorflow as tf
import numpy as np
import time

def test_gpu_training():
    """测试GPU训练"""
    print("=== GPU训练测试 ===")
    
    # 检查GPU
    gpus = tf.config.list_physical_devices('GPU')
    print(f"检测到 {len(gpus)} 个GPU设备")
    
    if not gpus:
        print("未检测到GPU，将使用CPU")
        return
    
    # 设置GPU内存增长
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"已为GPU {gpu.name} 启用内存增长")
        except RuntimeError as e:
            print(f"GPU {gpu.name} 内存增长设置失败: {e}")
    
    # 创建简单的测试数据
    print("\n创建测试数据...")
    X = np.random.random((1000, 100)).astype(np.float32)
    y = np.random.randint(0, 3, 1000).astype(np.int32)
    
    # 转换为one-hot编码
    y_onehot = tf.keras.utils.to_categorical(y, 3)
    
    # 创建简单模型
    print("创建模型...")
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(100,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"模型参数数量: {model.count_params():,}")
    
    # 训练测试
    print("\n开始训练测试...")
    start_time = time.time()
    
    history = model.fit(
        X, y,
        epochs=5,
        batch_size=32,
        verbose=1
    )
    
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"\n训练完成！")
    print(f"训练时间: {training_time:.2f} 秒")
    print(f"最终准确率: {history.history['accuracy'][-1]:.4f}")
    
    # 检查是否使用了GPU
    print(f"\n使用的设备:")
    print(f"模型设备: {model.layers[0].kernel.device}")
    
    # 测试预测
    print("\n测试预测...")
    predictions = model.predict(X[:10])
    print(f"预测形状: {predictions.shape}")
    print(f"预测样本: {predictions[0]}")
    
    print("\n=== GPU测试完成 ===")

if __name__ == "__main__":
    test_gpu_training() 