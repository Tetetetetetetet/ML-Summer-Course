import json
from pathlib import Path

# 1. 读取 JSON 文件
config_path = Path('config/feature.json')
with config_path.open(encoding='utf-8') as f:
    cfg = json.load(f)

# 2. 收集 iskeep == True 的特征名和描述
keep_lines = [
    f"{feat_name}: {meta.get('description', '')}"
    for feat_name, meta in cfg.get('features', {}).items()
    if meta.get('iskeep') is True
]

# 3. 写入 keep.txt
out_path = Path('keep.txt')
with out_path.open('w', encoding='utf-8') as f:
    f.write('\n'.join(keep_lines))

print(f"已写入 {len(keep_lines)} 条记录到 {out_path.resolve()}")