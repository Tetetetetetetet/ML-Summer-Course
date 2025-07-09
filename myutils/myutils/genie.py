import json
import os
from typing import Union
import pdb
import math
import re
from qwen_vl_utils import smart_resize
from transformers import AutoProcessor
MAX_PIXELS = 602112
def translate_action(response:str,dataset='android-lab'):
    '''
    translate assistant content to genie-format, according to `dataset`
    args:
    - response: item["messages"][-1]["content], e.g. tap(a3)
    return: dict, {"action": ,"args":}
    '''
    if dataset== 'android-lab':
        splits = re.split(r"[()]",response)
        action = splits[0]
        content = splits[1]
        if action=="tap":
            action = {"action":"tap","som":content}
        elif action=="finish":
            action = {"action":"terminate","status":"success"}
        elif action=="type":
            action = {"action":"type","text":content}
        elif action=="swipe":
            args = content.split(',')
            action = {"action":"swipe","direction":args[1].strip(),"dist":args[2].strip()}
        elif action=="back":
            action = {"action":"system_button","button":"Back"}
        else:
            raise ValueError(f'Unknown action type: {action}, from "{response}"')
    else:
        raise ValueError(f'Unknown dataset name: "{dataset}"')
    return action

def parse_action(value_str)->Union[dict,None]:
    try:
        if isinstance(value_str, str):
            value_str = value_str.strip()
            if '<tool_call>\n' in value_str:
                
                value_str = value_str.split('<tool_call>\n')[1].split('\n</tool_call>')[0]
            
            action = json.loads(value_str)
        else:
            action=value_str
        action = action['arguments']
        return action
    except Exception as e:
        print('error parsing action', value_str)
        match = re.search(r'(\{.*\})', value_str)
        if match:
            action_str = match.group(1)
            try:
                action = json.loads(action_str)
                return action
            except json.JSONDecodeError:
                return None
        else:
            return None

def resize_coord_back(x,y,width,height,processor,max_pixels=MAX_PIXELS):
    '''
    coord in image resized by processor -> coord in original image
    args:
    - x
    - y
    - height
    - processor
    - max_pixels: default=MAX_PIXELS
    returns:
    - x
    - y
    - (resized_width,resized_height)
    '''
    resized_height, resized_width  = smart_resize(height,
        width,
        factor=processor.image_processor.patch_size * processor.image_processor.merge_size,
        min_pixels=processor.image_processor.min_pixels,
        max_pixels=max_pixels)
    x = int(float(x/resized_width*width))
    y = int(float( y/resized_height*height))
    return x,y,(resized_width,resized_height)

def resize_coord(x,y,width,height,processor,max_pixels=MAX_PIXELS):
    resized_height, resized_width  = smart_resize(height,
        width,
        factor=processor.image_processor.patch_size * processor.image_processor.merge_size,
        min_pixels=processor.image_processor.min_pixels,
        max_pixels=max_pixels)
    x = int(float(x*resized_width/width))
    y = int(float( y*resized_height/height))
    return x,y,(resized_width,resized_height)

def parse_round_format_alab(data_item):
    '''
    convert a step/conversation with key="user | assistant" to genie format
    '''
    raw_text = data_item["messages"][0]["content"]
    raw_steps = re.split(r"Round \d+", raw_text)
    task_prompt = ""
    step_lines = []
    step_id = 1

    for segment in raw_steps:
        if "<|user|>" in segment and "<|assistant|>" in segment:
            user_query = re.search(r"<\|user\|>\n(.+?)\n", segment)
            assistant_action = re.search(r"<\|assistant\|>\n(.+)", segment)

            if user_query and not task_prompt:
                # 只用第一次的用户 query 作为任务目标
                task_prompt = user_query.group(1).strip()
            
            if assistant_action:
                action_text = assistant_action.group(1).strip()
                step_lines.append(f"Step{step_id}: {action_text}.")
                step_id += 1

    # 拼接 user 消息
    user_content = (
        f"The user query: {task_prompt}\n"
        f"Task progress (You have done the following operation on the current device): "
        + " ".join(step_lines)
    )

    # 获取最终 finish(...) 对应的动作
    response = data_item["messages"][-1]["content"]
    final_action = translate_action(response)
    final_content = {"name":"mobile_use","arguments":final_action}

    return {
        "messages": [
            {
                "role": "user",
                "content": user_content
            },
            {
                "role": "assistant",
                "content": "<tool_call>\n" + json.dumps(final_content, ensure_ascii=False) + "\n</tool_call>"
            }
        ],
        "images": data_item["images"]
    }
def calculate_f1_score(predicted_str, ground_truth_str):
    predicted_tokens = set(predicted_str.lower().split())
    ground_truth_tokens = set(ground_truth_str.lower().split())
    
    common_tokens = predicted_tokens.intersection(ground_truth_tokens)
    if len(predicted_tokens) == 0:
        precision = 0
    else:
        precision = len(common_tokens) / len(predicted_tokens)
    if len(ground_truth_tokens) == 0:
        recall = 0
    else:
        recall = len(common_tokens) / len(ground_truth_tokens)
    
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score
def determine_swipe_direction(start, end):
    try:
        x0, y0 = start
        x1, y1 = end
    except Exception as e:
        
        return "right"
    delta_x = x1 - x0
    delta_y = y1 - y0

    if abs(delta_x) > abs(delta_y):
        if delta_x > 0:
            return "right"
        else:
            return "left"
    else:
        if delta_y > 0:
            return "down"
        else:
            return "up"

def split_img_name(img_path:str):
    '''
    img_path->(ep_id,step_id)
    '''
    img_name = os.path.basename(img_path).split('.')[0].strip()
    ep_id = img_name.split('_')[0]
    step_id = img_name.split('_')[1]
    return ep_id,step_id
    
def compare_actions_ac(res:Union[dict,str,None], label:Union[dict,str,None], raw_id, processor,reverse_swipe=True, width=None, height = None): 
    type_equal = False
    if isinstance(res,str):
        res = parse_action(res)
    if isinstance(label,str):
        label = parse_action(label)
    if res is None:
        print('response="{}" cannot be parsed'.format(res))
    if label is None:
        print('label="{}" cannot be parsed'.format(label))
    if not res or not label:
        return False, False
    res_type = res.get('action', '').lower()
    label_type = label.get('action', '').lower()
    if res_type != label_type:
        return False, type_equal

    type_equal = True
    # if 'coordinate' in label and ('coordinate2' not in label) and (label_type!='swipe'):
    if ('click' in res_type) or ('long_press' in res_type):
        x, y = label['coordinate']
        
        assert height is not None
        assert width is not None
        resized_height, resized_width  = smart_resize(height,
            width,
            factor=processor.image_processor.patch_size * processor.image_processor.merge_size,
            min_pixels=processor.image_processor.min_pixels,
            max_pixels=processor.image_processor.max_pixels,)
        x = int(float(x/resized_width*width))
        y = int(float( y/resized_height*height))


        answer_xy = res.get('coordinate')

        if answer_xy is None:
            answer_x = 0
            answer_y = 0
        else:
            answer_x , answer_y = answer_xy

            
        answer_x = int(float(answer_x/resized_width*width))
        answer_y = int(float( answer_y/resized_height*height))
        try:
            answer_x = int(float(answer_x))
            answer_y = int(float(answer_y))
        except (TypeError, ValueError):
            return False, type_equal

        if math.sqrt((answer_x - x)**2 + (answer_y - y)**2) <= math.sqrt((width*0.14)**2 + (height*0.14)**2):
            return True, type_equal

        else:
            return False, type_equal

    else:
        if res_type == 'type':
            answer_text = res.get('text', '').strip().lower()
            gpt_text = label.get('text', '').strip().lower()
            answer_text = re.sub(r'\s+', ' ', answer_text)
            gpt_text = re.sub(r'\s+', ' ', gpt_text)
            if gpt_text in answer_text or answer_text in gpt_text:
                return True, type_equal
            else:
                # f1_score = calculate_f1_score(gpt_text, answer_text)
                # if f1_score >0.5:
                #     return True, selected_bound
            

                # else:
                return False, type_equal
        elif res_type == 'swipe':
            if 'direction' in res:
                answer_direction = res.get('direction', '')
            else:
                start_point = res.get('coordinate', '')
                end_point = res.get('coordinate2', '')
                if not isinstance(start_point, list):
                    # print('start_point', start_point)
                    answer_direction=end_point
                else:
                    answer_direction = determine_swipe_direction(start_point, end_point)
            gpt_direction = label.get('direction', '').strip().lower() or label.get('coordinate', '').strip().lower()
            if reverse_swipe:
                if 'up' in gpt_direction:
                    gpt_direction ='down'
                elif 'down' in gpt_direction:
                    gpt_direction='up'
                elif 'left' in gpt_direction:
                    gpt_direction='right'
                elif 'right' in gpt_direction:
                    gpt_direction='left'
            if gpt_direction in answer_direction or answer_direction in gpt_direction:
                return True, type_equal
            else:
                return False, type_equal
        elif res_type == 'open':
            answer_app_name = res.get('text', '').strip().lower()
            gpt_app_name = label.get('text', '').strip().lower()
            if gpt_app_name in answer_app_name or answer_app_name in gpt_app_name or (calculate_f1_score(answer_app_name, gpt_app_name)>0.5):
                return True, type_equal
            else:

                return False, type_equal
        elif res_type == 'terminate':
            answer_goal_status = res.get('status', '').strip().lower()
            gpt_goal_status = label.get('status', '').strip().lower()
            if gpt_goal_status in answer_goal_status or answer_goal_status in gpt_goal_status or (calculate_f1_score(answer_goal_status, gpt_goal_status)>0.5):
                return True, type_equal
            else:

                return False, type_equal
        elif res_type == 'system_button':
            answer_button = res.get('button', '').strip().lower()
            gpt_button = label.get('button', '').strip().lower()
            if gpt_button in answer_button or answer_button in gpt_button:
                return True, type_equal
            else:
                return False, type_equal
        elif res_type in [ 'wait']:
            return True, type_equal
        else:
            print('unrecognized action', res_type, res, label)
            return False, type_equal

def compare_action_androidlab(res:dict,label:dict,metrics:dict):
    '''
    check 2 action arguments e.g.{"action":"click","som":3,"action_desc":"..."}
    action_space = ["click","finish","type","swipe","back"]
    return
    - ele match
    - type match
    '''
    if label["action"] == "tap":
        label["action"] = "click"
    gt_act = label["action"]
    metrics["all_count"]+=1
    metrics[gt_act]["all_count"]+=1
    if res is None:
        metrics["format_error"]+=1
        return False,False
    if res["action"] != label["action"]:
        return False,False
    # type matches
    type_match = True
    ele_match = False
    act = res["action"]
    metrics["type_acc"]+=1
    metrics[act]["type_acc"]+=1
    if act == "click" and int(res["som"])==int(label["som"]):
        ele_match=True
    elif act == "finish" or act == "terminate": # no element acc
        metrics["acc"]+=1
        metrics[act]["acc"]+=1
    elif act == "type":
        answer_text = res.get('text', '').strip().lower()
        gpt_text = label.get('text', '').strip().lower()
        answer_text = re.sub(r'\s+', ' ', answer_text)
        gpt_text = re.sub(r'\s+', ' ', gpt_text)
        if gpt_text in answer_text or answer_text in gpt_text:
            ele_match=True
    elif act == "swipe" and res["direction"] == label["direction"] and res["dist"] == label["dist"]:
        ele_match=True
    elif act == "system_button" and res['button']==label['button']:
        ele_match=True
    else:
        return False,type_match
    if ele_match and type_match:
        metrics["acc"]+=1
        metrics[act]["acc"]+=1
        metrics["ele_acc"]+=1
        metrics[act]["ele_acc"]+=1
    return ele_match,type_match