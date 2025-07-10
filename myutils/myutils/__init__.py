import json
from pathlib import Path
import logging

logger = logging.getLogger(__file__)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s - %(name)s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def read_jsonl(meta_path):
    '''
    read a json/jsonl file
    Args:
        meta_path: str, the path to the json/jsonl file
    Returns:
        list of dicts/dict
    '''
    if meta_path.endswith('.jsonl'):
        meta_l = []
        with open(meta_path) as f:
            for i, line in enumerate(f):
                try:
                    meta_l.append(json.loads(line))
                except json.decoder.JSONDecodeError as e:
                    print(f"Error decoding the following jsonl line ({i}):\n{line.rstrip()}", flush=True)
                    raise e
    else:
        with open(meta_path, 'r') as f1:
            meta_l = json.load(f1)
    return meta_l

def write_jsonl(data, file_path:str):
    '''
    write data to a json/jsonl file
    Args:
        data: list of dicts or dict
        file_path: str, the path to the json/jsonl file
    Returns:
        None
    '''
    if file_path.endswith('.jsonl'):
        with open(file_path, "w", encoding="utf-8") as f:
            for item in data:
                json.dump(item, f, ensure_ascii=False)
                f.write("\n")
    else:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False,indent=4)

def rename_imgs(json_path:str,image_key:str="images"):
    '''
    for images in a json/jsonl, rename the image path relative to the .json file to absolute path
    Req:
    - the wanted image_key are directly accssible to each json obj in the file
    '''
    meta_data=read_jsonl(json_path)
    json_Path = Path(json_path)
    count = 0
    for item in meta_data:
        image_path = item[image_key][0]
        count +=len(item[image_key])
        image_path = json_Path.resolve().parent/Path(image_path)
        item[image_key] = [str(image_path.resolve())]
    output_path = json_Path.parent / ("re_"+json_Path.name)
    if output_path.name.endswith(".json"):
        output_path = output_path.parent / (output_path.stem+".jsonl")
    logger.info("output_path={}".format(output_path))
    input()
    with open(output_path,"w",encoding="utf-8") as f:
        for item in meta_data:
            json.dump(item,f,ensure_ascii=False)
            f.write('\n')
    logger.info("all image num = {}".format(count))

def dump_jsonl(data,json_path):
    if isinstance(data,dict):
        with open(json_path,"w",encoding="utf-8") as f:
            json.dump(data,f)
    else:
        with open(json_path,"w",encoding="utf-8") as f:
            for item in data:
                json.dump(item,f)
                f.write('\n')

def notnone(*args):
    '''
    Assert that all keyword arguments are not None.
    
    Usage:
        notnone(x=x, y=y, model=model)
    
    If any argument is None, raises AssertionError with the variable name.
    '''
    for val in args:
        assert val is not None, f"Variable '{val}' should not be None"
