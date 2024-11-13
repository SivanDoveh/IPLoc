import os
import numpy as np
os.environ['MKL_THREADING_LAYER'] = 'INTEL'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from peft import PeftModel
import math
import requests
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from transformers.image_utils import load_image
import argparse
import copy
from tqdm import tqdm
import json
import torchvision.ops as ops

from loc_dataset import get_dataloader
from utils_qwen import eval_bbox,pixel_to_qwen_format

import random 
random.seed(42)
import re
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import sys 
import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

DEVICE = "cuda:0"
def eval_model(args):
    dataloader = get_dataloader(args)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    iou_list=[]
    iou_bbox_list=[]

    boxes_preds_list=[]
    boxes_labels_list=[]

    results_dict = {
        "predictions": []
    }

    model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", torch_dtype=torch.bfloat16).cuda().eval()
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    if args.lora_weights_path:
        model = PeftModel.from_pretrained(model, args.lora_weights_path)
    
    print(f"{dataloader.__len__()}")
    for iii,data_item in enumerate(dataloader):
        torch.cuda.empty_cache()

        element,bbox,image_path,image_id,data = data_item
        element = element[0]

        messages = []
        question = f"<ref>{element}</ref>"
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"{image_path[0][0]}",
                    },
                    {"type": "text", "text": f"{question}"},
                ],
            }
        ]
        if bbox.__len__() > 1:
            for b_ind in range(bbox.__len__()-1):
                image = Image.open(image_path[b_ind][0])
                img_size = image.size
                box_norm = pixel_to_qwen_format(args,bbox[b_ind][0],img_size,'NotGT')

                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text","text": f"{box_norm}",
                            }],
                    })
                messages.append(
                                {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": f"{image_path[b_ind+1][0]}",
                            },
                            {"type": "text", "text": f"{question}"},
                        ],
                    }

                )
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(DEVICE)
        try:
            with torch.no_grad():
                try:
                    generated_ids = model.generate(**inputs, max_new_tokens=50)
                except:
                    for message in messages:
                        if "content" in message:
                            for content in message["content"]:
                                # For images
                                if "image" in content or "image_url" in content:
                                    content["resized_height"] = 224  # Set smaller height, e.g., 224 pixels
                                    content["resized_width"] = 224   # Set smaller width, e.g., 224 pixels
                                    content["min_pixels"] = 2 * 28 * 28  # Set smaller min pixels
                                    content["max_pixels"] = 8192 * 28 * 28  # Set smaller max pixels
                    image_inputs, video_inputs = process_vision_info(messages)
                    inputs = processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    )
                    inputs = inputs.to(DEVICE)
                    generated_ids = model.generate(**inputs, max_new_tokens=50)

            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            only_gen = output_text[0]
            GT = eval(bbox[-1][0])
            if eval_bbox(only_gen):
                query_image = load_image(image_path[-1][0])
                pred_box = eval_bbox(only_gen)
                pred_box = [pred_box[0][0], pred_box[0][1], pred_box[1][0], pred_box[1][1]]

                boxes_preds = torch.tensor(pred_box).unsqueeze(0)
                boxes_preds_list.append(pred_box)

                GT = pixel_to_qwen_format(args,bbox[-1][0],query_image.size,'GT')
                boxes_labels = torch.tensor(GT).unsqueeze(0)
                boxes_labels_list.append(GT)
                iou = ops.box_iou(boxes_preds, boxes_labels)
                iou_list.append(iou.item())
                iou_bbox_list.append(iou.item())

        except:
            print('error')

    return iou_list,iii+1,iou_bbox_list
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--curr_chunk", type=int, default=1)
    parser.add_argument("--chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--output_file", type=str, default="./outputs")
    parser.add_argument("--data_path", type=str, default="./Loc/data/path_to_test.json")
    parser.add_argument("--shots", type=int, default=2)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--lora_weights_path", type=str, default=None)
    parser.add_argument("--name", type=str, default="QwenVL2")

    args = parser.parse_args()
    args.data_name = (args.data_path.split('/')[-1]).split('.')[0]
    root = args.data_path.split(args.data_name)[0].split('data')[0]
    args.output_file = root+f'{args.name}_results/results_' + args.data_name+f'/results_data_{args.curr_chunk}.json'

    iou_list,samples,iou_bbox_list  = eval_model(args)

    dataloader = get_dataloader(args)
    # Convert tensor objects to serializable data types
    len_iou_bbox_list = len(iou_bbox_list)
    if len_iou_bbox_list == 0:
        len_iou_bbox_list = 1

    results_data = {
        "name":f"{args.data_name}",
        "iou_values": [iou for iou in iou_list],  
        "samples": samples,  
        "miou": sum(iou_list)/ dataloader.__len__() if iou_list else 0 ,
        "miou-bbox": sum(iou_bbox_list)/len_iou_bbox_list,
    }

    # Save the results in JSON format
    with open(args.output_file, 'w') as json_file:
        json.dump(results_data, json_file, indent=4)






    

