from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision.ops import box_iou
from PIL import Image, ImageDraw

def eval_bbox(bbox_str):
    try:
        start_idx = bbox_str.find('(')  # Find the first opening parenthesis
        if start_idx == -1:
            raise ValueError("No bounding box found")
    
        bbox_part = bbox_str[start_idx:].replace('(', '').replace(')', '')
        coords = bbox_part.split(',')
        x_min, y_min, x_max, y_max = map(int, coords)
        return (x_min, y_min), (x_max, y_max)
    except:
        try: 
            x_min, y_min, x_max, y_max = eval(bbox_str)
            return (x_min, y_min), (x_max, y_max) 
        except:
            return False

def pixel_to_qwen_format(args,bbox_str,img_size,state):
    if state!= 'GT' and "perseg" in args.data_path.split("/")[-1]:
        return eval(bbox_str)
    x_min, y_min,x_max, y_max = eval(bbox_str)
    
    img_width, img_height = img_size
    x_min_scaled = 1000 *(x_min / img_width)
    y_min_scaled = 1000 *(y_min / img_height)
    x_max_scaled = 1000 *(x_max / img_width)
    y_max_scaled = 1000 *(y_max / img_height)

    return [x_min_scaled,y_min_scaled,x_max_scaled,y_max_scaled]

def overlay_bbox(img, bbox, color="red"):
    x_min, y_min, x_max, y_max = bbox
    draw = ImageDraw.Draw(img)
    draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=10)
    return img

