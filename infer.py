import argparse
import os
import copy
 
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import json
import torch
from PIL import Image, ImageDraw, ImageFont
 
# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
 
# segment anything
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)
import cv2
import numpy as np
import matplotlib.pyplot as plt
 
 
def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image
 
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image
 
 
def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model
 
 
def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]
 
    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]
 
    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)
 
    return boxes_filt, pred_phrases
 
 
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
 
 
def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
    ax.text(x0, y0, label)
 
 
def save_mask_data(output_dir, mask_list, box_list, label_list, filename, H, W, label_map_dict):
    value = 0  # 0 for background
    # mask_img = torch.zeros(mask_list.shape[-2:])
    # mask_img = torch.zeros([H, W])
    mask_img = np.zeros([H, W], dtype=np.int32)
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    # plt.figure(figsize=(10, 10))
    # plt.imshow(mask_img.numpy())
    # plt.axis('off')
    # plt.savefig(os.path.join(output_dir, f'{filename}.png'), bbox_inches="tight", dpi=300, pad_inches=0.0)
    cv2.imwrite(os.path.join(output_dir, f'{filename}.png'), mask_img)
    # import pdb;pdb.set_trace()

    json_data = [{
        'value': value,
        'name': 'background',
        'label': 0
    }]
    for label, box in zip(label_list, box_list):
        value += 1 # instance
        name, logit = label.split('(')
        logit = logit[:-1]  # the last is ')'
        
        if name=='traffic _ cone': name='traffic_cone'
        # if name=='traffic cone': name='trafficcone'
        if ' ' in name and name!='traffic cone':
            name = name.split(' ')[0]
        if name == 'man':
            name = 'pedestrian'
        if name in ['traffic cone', 'cone', 'traffic']:
            name = 'traffic_cone'

        try:
            json_data.append({
                'value': value,
                'name': name,
                'label': label_map_dict[name],
                'logit': float(logit),
                'box': box.numpy().tolist(),
                'segmentation': []
            })
        except:
            print('name = ', name, box, filename)
            import pdb;pdb.set_trace()
    with open(os.path.join(output_dir, f'{filename}.json'), 'w') as f:
        json.dump(json_data, f)
 
 
if __name__ == "__main__":
 
    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, help="path to config file",
                        default='GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py')
    parser.add_argument(
        "--grounded_checkpoint", type=str, help="path to checkpoint file", default='groundingdino_swint_ogc.pth'
    )
    parser.add_argument(
        "--sam_version", type=str, default="vit_h", required=False, help="SAM ViT version: vit_b / vit_l / vit_h"
    )    
    parser.add_argument(
        "--sam_checkpoint", type=str, help="path to checkpoint file", default='sam_vit_h_4b8939.pth'
    )
    parser.add_argument(
        "--sam_hq_checkpoint", type=str, default=None, help="path to sam-hq checkpoint file"
    )
    parser.add_argument(
        "--use_sam_hq", action="store_true", help="using sam-hq for prediction"
    )    
    parser.add_argument("--input_dir", type=str,  help="path to input directory",default=r'E:\windmillvideo\test3')
    parser.add_argument("--text_prompt", type=str, help="text prompt", default='windmill')
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", help=r'D:\SAM-Tool\dataset\imgout'
    )
 
    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
    args = parser.parse_args()
    # cfg
    config_file = args.config  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_version = args.sam_version
    sam_checkpoint = args.sam_checkpoint
    sam_hq_checkpoint = args.sam_hq_checkpoint
    use_sam_hq = args.use_sam_hq
    input_dir = args.input_dir
    text_prompt = args.text_prompt
    output_dir = args.output_dir
    batch_size = args.batch_size
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    device = args.device
 
    import re
    s = 'barrier.bicycle.bus.car.motorcycle.pedestrian.traffic_cone.truck.road.sidewalk.terrain.vegetation.building.'
    items = re.split(r'[.]+', s)
    items = ['background'] + [item for item in items if item]

    label_map_dict = {items[i]:i for i in range(len(items))}

    # make dir
    os.makedirs(output_dir, exist_ok=True)
 
    # load model
    model = load_model(config_file, grounded_checkpoint, device=device)
    if use_sam_hq:
        predictor = SamPredictor(sam_hq_model_registry[sam_version](checkpoint=sam_hq_checkpoint).to(device))
    else:
        predictor = SamPredictor(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))    
    # iterate over input images
    input_files = [f for f in os.listdir(input_dir) if f.endswith('.jpg') or f.endswith('.png')]
    for idx, input_file in enumerate(input_files):
        print(f'Processing file {idx + 1} of {len(input_files)}: {input_file}')
        file_name = os.path.splitext(input_file)[0]
        todo_json_path = os.path.join(output_dir, f'{file_name}.json')
        if os.path.exists(todo_json_path):
            continue
        
        # load image
        image_pil, image = load_image(os.path.join(input_dir, input_file))
        # run grounding dino model
        boxes_filt, pred_phrases = get_grounding_output(
            model, image, text_prompt, box_threshold, text_threshold, device=device
        )
        # initialize SAM
        image = cv2.imread(os.path.join(input_dir, input_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)
 
        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]
 
        boxes_filt = boxes_filt.cpu()
 
        masks_list = []
        transformed_boxes_list = []
        for i in range(0, len(boxes_filt), batch_size):
            batch_boxes = boxes_filt[i:i + batch_size]
            transformed_boxes = predictor.transform.apply_boxes_torch(batch_boxes, image.shape[:2]).to(device)
            transformed_boxes_list.append(transformed_boxes)
            masks, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes.to(device),
                multimask_output=False,
            )
            masks_list.append(masks)
 
        transformed_boxes = torch.cat(transformed_boxes_list, dim=0)
        masks = torch.cat(masks_list, dim=0)
 
        # draw output image
        # plt.figure(figsize=(10, 10))
        # plt.imshow(image)
        # for mask in masks:
        #     show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        # for box, label in zip(boxes_filt, pred_phrases):
        #     show_box(box.numpy(), plt.gca(), label)
 
        # plt.axis('off')
        # plt.savefig(
        #     os.path.join(output_dir, f"grounded_sam_output_{input_file}"),
        #     bbox_inches="tight", dpi=300, pad_inches=0.0
        # )

        save_mask_data(output_dir, masks, boxes_filt, pred_phrases, os.path.splitext(input_file)[0], H, W, label_map_dict)
        #save_mask_data(output_dir, masks, boxes_filt, pred_phrases)
