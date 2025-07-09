#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修改feature.json文件，删除one-hot编码并以name作为key
"""
import json
from pathlib import Path

def modify_feature_json():
    """修改feature.json文件结构"""
    feature_file = Path(__file__).parent / "feature.json"
    
    # 读取原始feature.json
    with open(feature_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 创建新的结构
    new_data = {
        "dataset_name": data["dataset_name"],
        "target_feature": data["target_feature"],
        "features": {}
    }
    
    # 转换features数组为以name为key的字典
    for feature in data["features"]:
        feature_name = feature["name"]
        
        # 创建新的feature字典，删除one_hot_encoding
        new_feature = {
            "feature_id": feature["feature_id"],
            "category": feature["category"],
            "type": feature["type"],
            "description": feature["description"],
            "visualization": feature["visualization"]
        }
        
        # 保留label_encoding（如果存在）
        if "label_encoding" in feature:
            new_feature["label_encoding"] = feature["label_encoding"]
        
        # 以name作为key添加到features字典中
        new_data["features"][feature_name] = new_feature
    
    # 保存修改后的feature.json
    with open(feature_file, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)
    
    print("feature.json文件已修改完成！")
    print(f"特征数量: {len(new_data['features'])}")
    print("结构已改为以name作为key，并删除了one-hot编码信息")

if __name__ == "__main__":
    modify_feature_json() 