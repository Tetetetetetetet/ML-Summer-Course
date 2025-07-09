from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import os
import pdb
from typing import Union
from PIL import Image
import torch
from myutils import notnone
from qwen_vl_utils import process_vision_info

class Qwen:
    def __init__(self,model_name_or_path="/data1_hdd/wenbo/gui-agent/works/UI-Genie/src/ms-swift/ckpt/Qwen2.5-VL-3B-Instruct",max_pixels_choice=0,need_model:bool=True):
        self.model,self.processor = get_qwenvl(model_name_or_path,max_pixels_choice,need_model=need_model)

    def chat(self,prompt:Union[str,list],system_prompt:Union[str,None]=None, img_path=None,format=None,return_record=False,max_new_tokens=128,keep_assistant:bool=False):
        '''
        args:
            - prompt: str | gpt format conversation(s) | qwen format conversation(s)
            - format: 'gpt' | 'qwen' | None
        cases:
        1. prompt, img_path is str, format is None
        2. prompt, img_path is list[str], format is None
        2.5. prompt is str, img_path is list[str], format is None
        3. prompt is gpt format conversation(s) (dict | list[dict])
        4. prompt is qwen format conversation(s) (list[dict] | list[list[dict]])
        '''

        return chatqwenvl(self.model,self.processor,prompt,system_prompt,img_path,format,return_record,max_new_tokens,keep_assistant=keep_assistant)

def get_qwenvl(model_name_or_path="/data1_hdd/wenbo/gui-agent/works/UI-Genie/src/ms-swift/ckpt/Qwen2.5-VL-3B-Instruct",max_pixels_choice=0,need_model=True):
    if need_model:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto"
        )
    else:
        model = None
    mp_choice=[12845056,602112]
    if max_pixels_choice >= len(mp_choice):
        raise ValueError(f"max_pixels can be chosen in {mp_choice}, index {max_pixels_choice} out of range")
    processor = AutoProcessor.from_pretrained(model_name_or_path,max_pixels=mp_choice[max_pixels_choice])
    print(f"[INFO] processor information: patch size={processor.image_processor.patch_size},max pixels={processor.image_processor.max_pixels}, min pixels={processor.image_processor.min_pixels}")
    return model, processor

def gpt2qwen(conversations,keep_assitant:bool):
    '''
    transfer gpt format conversations e.g. [[{"message":{"role":str,"content":str},"images":[str]}]]
    -> qwen format conversations
    req:
    - support multiple round of conversations
    - support only one image input, will be placed in the first user message content
    '''
    qwen_format = {}
    if isinstance(conversations,dict):
        conversations = [conversations]
    convs = []
    for conversation in conversations:
        assert isinstance(conversation,dict)
        messages = conversation["messages"]
        user_m,sys_m,assistant_m=None,None,None
        conv = []
        image = conversation["images"][0]
        has_user = False
        for m in messages:
            if m["role"]=="user":
                if has_user:
                    user_m = {"role":"user","content":[{"type":"text","text":m["content"]}]}
                else:
                    user_m = {"role":"user","content":[{"type":"text","text":m["content"]},{"type":"image","image":image}]}
                    has_user=True
                conv.append(user_m)
            elif m["role"]=='system':
                sys_m = {"role":"system","content":[{"type":"text","text":m["content"]}]}
                conv.append(sys_m)
            elif m["role"]=="assistant" and keep_assitant:
                assistant_m = {"role":"assistant","content":[{"type":"text","text":m["content"]}]}
                conv.append(assistant_m)
            else:
                raise ValueError("Unknown role: {}".format(m["role"]))
        convs.append(conv)
    return convs
            
def build_qwen_conv_with_text(prompt:str,system_prompt=None,img_paths:Union[list,str,None]=None):
    sys_m = {"role":"system","content":[{"type":"text","text":system_prompt}]}
    user_m = {"role":"user","content":[{"type":"text","text":prompt}]}
    if img_paths is not None:
        if isinstance(img_paths,str):
            img_paths = [img_paths]
        for img in img_paths:
            user_m["content"].append({"type":"image","image":img})
    if system_prompt !=None:
        conv = [sys_m,user_m]
    else:
        conv = [user_m]
    return conv

def chatqwenvl(model, processor, prompt:Union[str,list],system_prompt:Union[str,None]=None, img_path=None,format=None,return_record=False,max_new_tokens=128,keep_assistant:bool=False):
    '''
    args:
    - prompt: str | gpt format conversation(s) | qwen format conversation(s)
    - format: 'gpt' | 'qwen' | None
    cases:
    1. prompt, img_path is str, format is None
    2. prompt, img_path is list[str], format is None
    2.5. prompt is str, img_path is list[str], format is None
    3. prompt is gpt format conversation(s) (dict | list[dict])
    4. prompt is qwen format conversation(s) (list[dict] | list[list[dict]])
    '''
    record={}
    # build into qwen format conversations
    if format == None:
        convs = []
        if isinstance(prompt,str):
            convs = [build_qwen_conv_with_text(prompt,system_prompt,img_path)] #type: ignore
        elif isinstance(prompt,list) and isinstance(img_path,list):
            system_prompts = [system_prompt for _ in prompt]
            for p,(s,img) in zip(prompt,zip(system_prompts,img_path)):
                conv = build_qwen_conv_with_text(p,s,img)
                convs.append(conv)
        else:
            raise ValueError("Cannot process input: prompt:{}, format:{}, system_prompt:{}".format(prompt,format,system_prompt))
    elif format == "gpt" and (isinstance(prompt,dict) or isinstance(prompt,list)):
        convs = gpt2qwen(prompt,keep_assitant=keep_assistant)
    elif format == "qwen":
        if isinstance(prompt[0],dict):
            convs = [prompt]
        else:
            convs = prompt
    else:
        raise ValueError("Cannot process input: prompt:{}, format:{}, system_prompt:{}".format(prompt,format,system_prompt))
    record["conversations"]=convs
    # conversations -> completions
    texts = [processor.apply_chat_template(conv,tokenize=False,add_generation_prompt=True) for conv in convs]
    record["input_text"]=texts
    assert convs is not None and isinstance(convs,list)
    image_inputs,video_inputs = process_vision_info(convs) #type: ignore
    # check for pixels
    all_pix=0
    assert image_inputs is not None
    for image in image_inputs:
        pix = image.size[0]*image.size[1]
        all_pix+=pix
    # prepare inputs
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True
    )
    record["inputs"]=inputs
    inputs = inputs.to('cuda')
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    full_text = processor.batch_decode(
        generated_ids,skip_special_tokens=True,clean_up_tokenization_spaces=False
    )
    record['text']=full_text
    if return_record:
        return output_text,record
    else:
        return output_text
    

def exp():
    max_new_tokens=1024
    # model,processor = get_qwenvl("/data1_hdd/wenbo/gui-agent/works/UI-Genie/src/ms-swift/ckpt/Qwen2.5-VL-3B-Instruct",max_pixels_choice=0)
    model,processor = get_qwenvl("Qwen/Qwen2.5-VL-7B-Instruct",max_pixels_choice=1)
    img_path = "/data1_hdd/wenbo/gui-agent/works/UI-Genie/src/ms-swift/current_img.png"
    img_paths = ["../image.png", "../image2.png","used_img.png"]
    res = chatqwenvl(model,processor,prompt="You were given multiple images. Even if two images are exactly the same, you should count them as separate pictures. Please answer in the format <answer>X</answer>.",img_path=img_paths,max_new_tokens=max_new_tokens)
    # res = chatqwenvl(model,processor,"How many picture do I give you? Give me answer in the format <answer>number</answer>",img_path=img_path)
    print(res)
    res = chatqwenvl(model,processor,prompt="You were given multiple images, recognize what they are and answer me in the format of <answer>1....,2....../answer>, for example <answer>1.cat,2.dot,3.xxx</answer>",img_path=img_paths,max_new_tokens=max_new_tokens)
    print(res)
    
if __name__=="__main__":
    exp()