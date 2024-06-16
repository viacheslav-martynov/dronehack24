"""
Here we crop objects by their gt bboxes
to form classification dataset

"""
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
from PIL import Image, ImageDraw
from fire import Fire

def get_object_classes(image_labels):
    object_classes = []
    for image_label in image_labels:
        object_classes.append(image_label[0])
    return object_classes

def get_image_labels(image_labels_path):
    with open(image_labels_path, 'r') as f:
        image_labels = []
        for image_label in f.readlines():
            image_labels.append(list(map(float, image_label.strip().split())))
    return image_labels

def get_boxes_pixel_xyxy(image_labels, image_size):
    """
    Computes pixel box size of objects
    """
    shapes = []
    for image_label in image_labels:
        shape = [
            (image_label[1]-image_label[3]/2)*image_size[0],
            (image_label[2]-image_label[4]/2)*image_size[1],
            (image_label[1]+image_label[3]/2)*image_size[0],
            (image_label[2]+image_label[4]/2)*image_size[1]
        ]
        shapes.append(shape)
    return shapes

def get_objects_for_crop(boxes_pixel_xyxy, min_pixel_size, object_classes):
    """
    Computes pixel box size of objects in one image 
    and returns objects that must be cropped and resized.
    """
    boxes_to_scale = []
    for box_pixel_xyxy, object_class in zip(boxes_pixel_xyxy, object_classes):
        k_w_min = 1
        k_h_min = 1
        width = box_pixel_xyxy[2] - box_pixel_xyxy[0]
        height = box_pixel_xyxy[3] - box_pixel_xyxy[1]

        if (width <= min_pixel_size) | (height <= min_pixel_size):
            return []

        # if width < min_pixel_size:
        #     # Minimal coef for scaling witdth to match minimal requirement
        #     k_w_min = min_pixel_size / width
        # if height < min_pixel_size:
        #     # Minimal coef for scaling witdth to match minimal requirement
        #     k_h_min = min_pixel_size / height

        scale_coef = 10#max(k_w_min, k_h_min)
        if scale_coef>1:
            scale_coef *= np.random.uniform(1, 1.25) 
            boxes_to_scale.append((box_pixel_xyxy, scale_coef, object_class))
    return boxes_to_scale

def get_new_labels(crop, image_labels, object_classes):
    """
    crop  - xyxy
    image_labels  - xyxy
    """

    crop_width = crop[2]-crop[0]
    crop_height = crop[3]-crop[1]
    shifted_labels = []
    for image_label, object_class in zip(image_labels, object_classes):
        shifted_label = [
            image_label[0]-crop[0], #minx
            image_label[1]-crop[1], #miny
            image_label[2]-crop[0], #maxx
            image_label[3]-crop[1]  #maxy
        ]
        # cases to save shifted label (if it is not outside the crop)
        if not ( \
            (shifted_label[0] <= 0) or \
            (shifted_label[1] <= 0) or \
            (shifted_label[2] >= crop_width) or \
            (shifted_label[3] >= crop_height)
            ):
            shifted_labels.append(
                [object_class] + list(xyxy2yolo(
                    shifted_label, (crop_width, crop_height))))
            
    return shifted_labels

def xyxy2yolo(xyxy, image_size):
    x_center = (xyxy[0] + xyxy[2]) / 2 / image_size[0]
    y_center = (xyxy[1] + xyxy[3]) / 2 / image_size[1]
    width = (xyxy[2] - xyxy[0])/ image_size[0]
    height = (xyxy[3] - xyxy[1])/ image_size[1]
    return x_center, y_center, width, height

def yolo2xyxy(yolo, image_size):
    return ((yolo[0]-yolo[2]/2)*image_size[0],
            (yolo[1]-yolo[3]/2)*image_size[1],
            (yolo[0]+yolo[2]/2)*image_size[0],
            (yolo[1]+yolo[3]/2)*image_size[1])

def bounded_normal(mu, sigma, lower_bound, upper_bound):
    while True:
        value = np.random.normal(mu, sigma)
        if lower_bound <= value <= upper_bound:
            return value

def crop_objects(image, image_labels, object_classes, objects_for_crop, random=True):
    image_width, image_height = image.size
    image_aspect_ratio = image_width/image_height
    crops = []
    crops_new_labels = []
    for object_for_crop in objects_for_crop:
        obj_class = object_for_crop[2]
        scale = object_for_crop[1]
        xyxy = object_for_crop[0]
        width = xyxy[2]-xyxy[0]
        height = xyxy[3]-xyxy[1]
        x_center = int((xyxy[2]+xyxy[0])/2)
        y_center = int((xyxy[3]+xyxy[1])/2)

        crop_height = height*np.random.uniform(0.95,1.25)
        crop_width = width*np.random.uniform(0.95,1.25)
    

        x_center_coef = 0
        y_center_coef = 0
        if random:
            #x_center_coef = np.random.uniform(-0.55,0.55)*crop_width/2
            #y_center_coef = np.random.uniform(-0.55,0.55)*crop_height/2
            sigma = np.random.uniform(0.05,0.15)
            x_center_coef = bounded_normal(mu = 0, sigma = sigma, lower_bound = -1, upper_bound = 1) * crop_width / 2  # Гауссово распределение
            y_center_coef = bounded_normal(mu = 0, sigma = sigma, lower_bound = -1, upper_bound = 1) * crop_height / 2

        x_min = x_center+x_center_coef - crop_width/2
        x_max = x_center+x_center_coef + crop_width/2
        y_min = y_center+y_center_coef - crop_height/2
        y_max = y_center+y_center_coef + crop_height/2

        crop = image.crop((x_min, y_min, x_max, y_max))
        # new_labels = get_new_labels([x_min, y_min, x_max, y_max], image_labels, object_classes)
        crops.append(crop)
        crops_new_labels.append(obj_class)
    return crops, crops_new_labels

def save_crop(target_dir, name, crop_image, crop_labels):
    if target_dir is str:
        target_dir = Path(target_dir)
    name = Path(name)
    # create directories
    os.makedirs(Path(target_dir) / Path(str(int(crop_labels))), exist_ok=True)
    # os.makedirs(Path(target_dir) / Path('labels'), exist_ok=True)
    # save image
    crop_image.save(Path(target_dir) / Path(str(int(crop_labels))) / name)
    # # save labels
    # with open(Path(target_dir) / Path('labels') / name.with_suffix('.txt'), 'w') as f:
    #     f.write(str(crop_labels)+'\n')
        
  
def main(base_dir: str, save_dir: str, min_pixel_size: int = 20, log_file: str = "empty_images.txt"):
    if Path(base_dir).is_file():
        with open(base_dir, 'r') as file:
            image_paths = [Path(line.strip()) for line in file.readlines()]
            labels_dir = image_paths[1].parent.parent / Path("labels")
    else:
        base_dir = Path(base_dir)
        imgs_dir = base_dir / Path("images")
        labels_dir = base_dir / Path("labels")
        image_paths = [imgs_dir / filename for filename in imgs_dir.rglob("*")]

    with open(log_file, 'w') as log:
        for image_path in tqdm(image_paths):
            image_filename = image_path.stem
            image_suffix = image_path.suffix
            if image_suffix != '.json':
                # Get image
                image = Image.open(image_path)
                # Get labels
                image_labels_path = labels_dir / f'{image_path.stem}.txt'
                image_labels = get_image_labels(image_labels_path)
                object_classes = get_object_classes(image_labels)

                # Get objects for crop
                boxes_pixel_xyxy = get_boxes_pixel_xyxy(image_labels, image.size)

                objects_for_crop = get_objects_for_crop(boxes_pixel_xyxy, min_pixel_size, object_classes)
                crops, crops_new_labels = crop_objects(image, boxes_pixel_xyxy, object_classes, objects_for_crop)
                if not crops:
                    log.write(str(image_path) + '\n')
                for i, (crop, crop_new_labels) in enumerate(zip(crops, crops_new_labels)):
                    save_crop(
                        save_dir,
                        f"{image_filename}_crop_{i}{image_suffix}",
                        crop_image=crop,
                        crop_labels=crop_new_labels)

if __name__ == '__main__':
    Fire(main)