#!/usr/bin/env python3
"""
糖尿病数据集分析项目 - 环境测试脚本
"""

import sys
import os
from pathlib import Path

def test_imports():
    """测试所有必要的包导入"""
    print("=== 测试包导入 ===")
    
    try:
        import pandas as pd
        print("✓ pandas 导入成功")
    except ImportError as e:
        print(f"✗ pandas 导入失败: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ numpy 导入成功")
    except ImportError as e:
        print(f"✗ numpy 导入失败: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✓ matplotlib 导入成功")
    except ImportError as e:
        print(f"✗ matplotlib 导入失败: {e}")
        return False
    
    try:
        import seaborn as sns
        print("✓ seaborn 导入成功")
    except ImportError as e:
        print(f"✗ seaborn 导入失败: {e}")
        return False
    
    try:
        import sklearn
        print("✓ scikit-learn 导入成功")
    except ImportError as e:
        print(f"✗ scikit-learn 导入失败: {e}")
        return False
    
    try:
        import myutils
        print("✓ myutils 导入成功")
    except ImportError as e:
        print(f"✗ myutils 导入失败: {e}")
        return False
    
    return True

def test_data_files():
    """测试数据文件是否存在"""
    print("\n=== 测试数据文件 ===")
    
    dataset_path = Path("Dataset")
    required_files = [
        "diabetic_data_training.csv",
        "diabetic_data_test.csv",
        "IDS_mapping.csv"
    ]
    
    all_exist = True
    for file in required_files:
        file_path = dataset_path / file
        if file_path.exists():
            print(f"✓ {file} 存在")
        else:
            print(f"✗ {file} 不存在")
            all_exist = False
    
    return all_exist

def test_source_files():
    """测试源代码文件是否存在"""
    print("\n=== 测试源代码文件 ===")
    
    src_path = Path("src")
    required_files = [
        "data_visualization.py"
    ]
    
    # 检查feature.json（在config目录）
    feature_json_path = Path("config/feature.json")
    if feature_json_path.exists():
        print("✓ feature.json 存在")
    else:
        print("✗ feature.json 不存在")
        all_exist = False
    
    all_exist = True
    for file in required_files:
        file_path = src_path / file
        if file_path.exists():
            print(f"✓ {file} 存在")
        else:
            print(f"✗ {file} 不存在")
            all_exist = False
    
    return all_exist

def test_myutils():
    """测试myutils包功能"""
    print("\n=== 测试myutils功能 ===")
    
    try:
        from myutils import read_jsonl
        print("✓ read_jsonl 函数可用")
    except Exception as e:
        print(f"✗ read_jsonl 函数不可用: {e}")
        return False
    
    try:
        from myutils import write_jsonl
        print("✓ write_jsonl 函数可用")
    except Exception as e:
        print(f"✗ write_jsonl 函数不可用: {e}")
        return False
    
    return True

def test_data_loading():
    """测试数据加载功能"""
    print("\n=== 测试数据加载 ===")
    
    try:
        import pandas as pd
        from pathlib import Path
        
        # 测试训练集加载
        train_path = Path("Dataset/diabetic_data_training.csv")
        if train_path.exists():
            train_data = pd.read_csv(train_path)
            print(f"✓ 训练集加载成功: {train_data.shape}")
        else:
            print("✗ 训练集文件不存在")
            return False
        
        # 测试测试集加载
        test_path = Path("Dataset/diabetic_data_test.csv")
        if test_path.exists():
            test_data = pd.read_csv(test_path)
            print(f"✓ 测试集加载成功: {test_data.shape}")
        else:
            print("✗ 测试集文件不存在")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        return False

def main():
    """主测试函数"""
    print("糖尿病数据集分析项目 - 环境测试")
    print("=" * 50)
    
    # 检查当前目录
    current_dir = Path.cwd()
    print(f"当前目录: {current_dir}")
    
    # 运行所有测试
    tests = [
        test_imports,
        test_data_files,
        test_source_files,
        test_myutils,
        test_data_loading
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ 测试执行失败: {e}")
            results.append(False)
    
    # 总结
    print("\n" + "=" * 50)
    print("测试总结:")
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✓ 所有测试通过 ({passed}/{total})")
        print("项目环境配置正确，可以开始使用！")
        return 0
    else:
        print(f"✗ 部分测试失败 ({passed}/{total})")
        print("请检查环境配置和文件完整性")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 