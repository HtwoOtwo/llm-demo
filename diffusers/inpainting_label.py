import argparse
import os
import random
from pathlib import Path
from typing import Union

import cv2
import gradio as gr
import numpy as np
import pycocotools.mask as maskUtils
from PIL import Image


def uncenter(x):
    assert (
        x.shape[-1] == 4
    ), f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = np.empty_like(x)  # faster than clone/copy
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 2]  # w
    y[..., 3] = x[..., 3]  # h
    return y


def xyxyn2xyxy(x, w=640, h=640, padw=0, padh=0):
    y = np.empty_like(x)
    y[..., 0::2] = x[..., 0::2] * w
    y[..., 1::2] = x[..., 1::2] * h
    return y


def get_bboxes_xywh(x):
    y = np.array([0.0, 0.0, 0.0, 0.0])
    y[0] = np.min(x[..., 0::2])
    y[1] = np.min(x[..., 1::2])
    y[2] = np.max(x[..., 0::2]) - np.min(x[..., 0::2])
    y[3] = np.max(x[..., 1::2]) - np.min(x[..., 1::2])
    return y


def poly2mask(mask_ann: Union[list, dict], img_w: int, img_h: int) -> np.ndarray:
    """Private function to convert masks represented with polygon to
    bitmaps.

    Args:
        mask_ann (list | dict): Polygon mask annotation input.
        img_h (int): The height of output mask.
        img_w (int): The width of output mask.

    Returns:
        np.ndarray: The decode bitmap mask of shape (img_h, img_w).
    """

    if isinstance(mask_ann, list):
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        rle = maskUtils.merge(rles)
    elif isinstance(mask_ann["counts"], list):
        # uncompressed RLE
        rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
    else:
        # rle
        rle = mask_ann
    mask = maskUtils.decode(rle)
    return mask

def is_clockwise(contour):
    value = 0
    num = len(contour)
    for i, point in enumerate(contour):
        p1 = point
        if i < num - 1:
            p2 = contour[i + 1]
        else:
            p2 = contour[0]
        value += (p2[0][0] - p1[0][0]) * (p2[0][1] + p1[0][1])
    return value < 0


def get_merge_point_idx(contour1, contour2):
    idx1 = 0
    idx2 = 0
    distance_min = -1
    for i, p1 in enumerate(contour1):
        for j, p2 in enumerate(contour2):
            distance = pow(p2[0][0] - p1[0][0], 2) + pow(p2[0][1] - p1[0][1], 2)
            if distance_min < 0:
                distance_min = distance
                idx1 = i
                idx2 = j
            elif distance < distance_min:
                distance_min = distance
                idx1 = i
                idx2 = j
    return idx1, idx2


def merge_contours(contour1, contour2, idx1, idx2):
    contour = []
    for i in list(range(0, idx1 + 1)):
        contour.append(contour1[i])
    for i in list(range(idx2, len(contour2))):
        contour.append(contour2[i])
    for i in list(range(0, idx2 + 1)):
        contour.append(contour2[i])
    for i in list(range(idx1, len(contour1))):
        contour.append(contour1[i])
    contour = np.array(contour)
    return contour


def merge_with_parent(contour_parent, contour):
    if not is_clockwise(contour_parent):
        contour_parent = contour_parent[::-1]
    if is_clockwise(contour):
        contour = contour[::-1]
    idx1, idx2 = get_merge_point_idx(contour_parent, contour)
    return merge_contours(contour_parent, contour, idx1, idx2)



def mask2polygon(mask):
    contours, hierarchies = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    contours_approx = []
    polygons = []
    for contour in contours:
        epsilon = 0.001 * cv2.arcLength(contour, True)
        contour_approx = cv2.approxPolyDP(contour, epsilon, True)
        contours_approx.append(contour_approx)

    contours_parent = []
    for i, contour in enumerate(contours_approx):
        parent_idx = hierarchies[0][i][3]
        if parent_idx < 0 and len(contour) >= 3:
            contours_parent.append(contour)
        else:
            contours_parent.append([])

    for i, contour in enumerate(contours_approx):
        parent_idx = hierarchies[0][i][3]
        if parent_idx >= 0 and len(contour) >= 3:
            contour_parent = contours_parent[parent_idx]
            if len(contour_parent) == 0:
                continue
            contours_parent[parent_idx] = merge_with_parent(contour_parent, contour)

    contours_parent_tmp = []
    for contour in contours_parent:
        if len(contour) == 0:
            continue
        contours_parent_tmp.append(contour)

    polygons = []
    for contour in contours_parent_tmp:
        polygon = contour.flatten().tolist()
        polygons.append(polygon)
    return polygons



def process_annotation(label_path, w, h) -> dict:
    with open(label_path) as f:
        lines = f.readlines()
        clss = [int(line.split(" ")[0]) for line in lines]
        anns = [[float(e) for e in line.rstrip("\n").split(" ")[1:]] for line in lines]
        assert len(clss) == len(anns)
    if next((ann for ann in anns if len(ann) > 4), None) is not None:
        # denormalize
        anns = [process_mask(xyxyn2xyxy(np.array(ann), w, h), w, h) for ann in anns]
    else:
        raise ValueError("Unrecognized ann.")
    return anns


def process_mask(ann, mask_w, mask_h):
    # polygon
    if isinstance(ann, list) or isinstance(ann, np.ndarray):
        if all(isinstance(item, list) for item in ann):
            polygons = [
                np.array(polygon)
                for polygon in ann
                if len(polygon) % 2 == 0 and len(polygon) >= 6
            ]
            if len(polygons) == 0:
                polygons = [np.zeros(6)]
            gt_mask = poly2mask(polygons, mask_w, mask_h)
        else:
            if len(ann) % 2 == 0 and len(ann) >= 6:
                polygons = [np.array(ann)]
            else:
                polygons = [np.zeros(6)]
            gt_mask = poly2mask(polygons, mask_w, mask_h)
    # rle, convert to polygon
    elif isinstance(ann, dict) and (
        ann.get("counts") is not None
        and ann.get("size") is not None
        and isinstance(ann["counts"], (list, str))
    ):
        segmentation = maskUtils.frPyObjects(ann, *ann["size"])
        gt_mask = maskUtils.decode(segmentation)
        gt_mask[gt_mask > 0] = 255
    else:
        raise ValueError("Unrecognized ann.")
    return gt_mask


def draw_mask(image, mask, color):
    overlay = image.copy()
    colored_mask = np.zeros_like(image)

    for i in range(3):
        colored_mask[..., i] = mask * color[i]
    
    overlay = cv2.addWeighted(overlay, 1, colored_mask, 0.5, 0)

    return overlay

def parse_args():
    parser = argparse.ArgumentParser(description="Inpainting labeling tool")
    parser.add_argument(
        "--save_folder",
        type=str,
        default=None,
        required=True,
        help="Path to save folder",
    )
    
    return parser.parse_args()
    
class ProcessManager:
    def __init__(self):
        self.prompt_map = {}  # id: prompt
        self.mask_map = {}
        self.masks_color = []
        
    def upload(self, image_path):
        # reset
        self.prompt_map = {}
        self.mask_map = {}
        self.masks_color = []
        return Image.open(image_path) if image_path is not None else None
        
        
    def annotate_image(self, image, label_path):
        if label_path is None:
            mask = image["layers"][0][:,:,-1]
            image = image["background"]

            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            while color in self.masks_color:
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            
            self.masks_color.append(color)
            masked_image = draw_mask(image, mask, color)
        else:
            image = image["background"]
            h, w = image.shape[:2]
            masks = process_annotation(label_path, w, h)

            num_masks = len(masks)
            self.masks_color = [
                (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                for _ in range(num_masks)
            ]
            self.mask_map = {i: mask for i, mask in enumerate(masks) }
            
            mask_0, color_0 = self.mask_map[0], self.masks_color[0]
            masked_image = draw_mask(image, mask_0, color_0)
            
        return masked_image, len(self.prompt_map), len(self.prompt_map)


    def show_record_prompt(self, slider, image):        
        try:
            overlay = draw_mask(image["background"], self.mask_map[slider], self.masks_color[slider])
        except Exception:
            overlay = None
        prompt = self.prompt_map.get(slider, "")
        
        return prompt, slider, overlay
    
    def save_mask(self, image, num, prompt, file_upload):
        file_upload = Path(file_upload)
        image_name = file_upload.stem
        save_folder = Path(DATA_ROOT) / image_name
        
        mask = image["layers"][0][:,:,-1]
        image = image["background"]
        
        # record
        if prompt.strip():  # Ensure non-empty prompts are recorded
            if num not in self.prompt_map:
                self.prompt_map[num] = prompt
            if num not in self.mask_map:
                self.mask_map[num] = mask
            print(f"Recorded {num} prompt: {prompt}")
            
            mask_path = save_folder / f"{prompt}.png"
            image_path = save_folder / f"{image_name}.png"
            os.makedirs(save_folder, exist_ok=True)
            Image.fromarray(self.mask_map[num]).save(mask_path)
            if not os.path.exists(image_path):
                Image.fromarray(image).save(image_path)
        
        return num + 1, num + 1, None, image
    

manager = ProcessManager()

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            file_upload = gr.File(label="Upload Image")
            label_input = gr.File(label="Upload Labels, optional, please use yolo format txt file.")
            image_input = gr.ImageMask(label="Image display", layers=False)
            process_btn1 = gr.Button("Annotate")

        with gr.Column():
            image_annotated = gr.Image(label="Processed Image")
            slider = gr.Slider(0, 100, step=1, label="Slider")
            with gr.Row():
                num = gr.Number(label="mask id") # record from where the slider is
                prompt = gr.Textbox(placeholder="Prompt")
            process_btn2 = gr.Button("Save")
            process_btn3 = gr.Button("Undo") # TODO

    # Define the event listeners
    file_upload.change(fn=manager.upload, inputs=[file_upload], outputs=[image_input])
    process_btn1.click(
        fn=manager.annotate_image, inputs=[image_input, label_input], outputs=[image_annotated, slider, num]
    )
    process_btn2.click(fn=manager.save_mask, inputs=[image_input, num, prompt, file_upload], outputs=[slider, num, image_annotated, image_input])
    slider.change(manager.show_record_prompt, inputs=[slider, image_input], outputs=[prompt, num, image_annotated])

# Launch the app
args = parse_args()
DATA_ROOT = args.save_folder
demo.launch(show_error=True)
