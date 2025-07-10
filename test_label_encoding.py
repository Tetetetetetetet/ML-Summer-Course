#!/usr/bin/env python3
"""
测试label_encoding一致性的脚本
"""

def test_label_encoding_consistency():
    """测试label_encoding一致性的示例"""
    
    # 测试用例1: 正确的一致顺序
    correct_encoding = {
        'unique_values': ['A', 'B', 'C'],
        'encoding_mapping': {'A': 0, 'B': 1, 'C': 2}
    }
    
    # 测试用例2: 不一致的顺序
    incorrect_encoding = {
        'unique_values': ['A', 'B', 'C'],
        'encoding_mapping': {'A': 0, 'C': 1, 'B': 2}  # B和C的顺序不对
    }
    
    def verify_consistency(label_encoding):
        """验证label_encoding的一致性"""
        unique_values = label_encoding['unique_values']
        encoding_mapping = label_encoding['encoding_mapping']
        
        for i, value in enumerate(unique_values):
            if str(value) not in encoding_mapping or encoding_mapping[str(value)] != i:
                return False, f"Index {i}: unique_values[{i}]={value}, encoding_mapping['{value}']={encoding_mapping.get(str(value), 'NOT_FOUND')}"
        return True, "Consistent"
    
    # 测试正确的情况
    is_consistent, message = verify_consistency(correct_encoding)
    print(f"Correct encoding: {'✓' if is_consistent else '✗'} {message}")
    
    # 测试错误的情况
    is_consistent, message = verify_consistency(incorrect_encoding)
    print(f"Incorrect encoding: {'✓' if is_consistent else '✗'} {message}")
    
    # 测试修复函数
    def fix_consistency(label_encoding):
        """修复label_encoding的一致性"""
        encoding_mapping = label_encoding['encoding_mapping']
        if not encoding_mapping:
            return label_encoding
            
        # 创建从编码到值的反向映射
        reverse_mapping = {v: k for k, v in encoding_mapping.items()}
        
        # 按编码顺序重新排列unique_values
        sorted_values = []
        for i in range(len(encoding_mapping)):
            if str(i) in reverse_mapping:
                sorted_values.append(reverse_mapping[str(i)])
        
        label_encoding['unique_values'] = sorted_values
        return label_encoding
    
    # 修复不一致的编码
    fixed_encoding = fix_consistency(incorrect_encoding.copy())
    is_consistent, message = verify_consistency(fixed_encoding)
    print(f"Fixed encoding: {'✓' if is_consistent else '✗'} {message}")
    print(f"Fixed unique_values: {fixed_encoding['unique_values']}")

if __name__ == "__main__":
    test_label_encoding_consistency() 