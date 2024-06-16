import json
import os
from fire import Fire
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Set the paths for the input and output directories

# The directory with images and labels subdirectories

def main(input_images_filename: str, output_path: str = '.'):
    """
    Converts yolo styled annotations to coco format.

    Arguments
    ---------
    input_dir : str
        input directory with images and labels subdirectories
    output_dir : str
        directory to save coco annotations
    """

    # Define the categories for the COCO dataset
    categories = [
        {"id": 0, "name": "copter_type_uav"},
        {"id": 1, "name": "aircraft"},
        {"id": 2, "name": "helicopter"},
        {"id": 3, "name": "bird"},
        {"id": 4, "name": "aircraft_type_uav"}]

    # Define the COCO dataset dictionary
    coco_dataset = {
        "info": {},
        "licenses": [],
        "categories": categories,
        "images": [],
        "annotations": []
    }

    annos_id = 0

    with open(input_images_filename, 'r') as f:
         image_paths = [Path(filename.strip()) for filename in f.readlines()]

    for image_id, image_path in tqdm(
        enumerate(image_paths),
        total=len(image_paths)
    ):
        image_name = Path(image_path.name)
        labels_dir = Path(*image_path.parts[:-2], 'labels')
        label_path = labels_dir / f'{image_name.stem}.txt'

        image = Image.open(image_path)
        width, height = image.size
        
        # Add the image to the COCO dataset
        image_dict = {
            "id": image_id, #int(image_file.split('.')[0]),
            "width": width,
            "height": height,
            "file_name": str(image_path)
        }
        coco_dataset["images"].append(image_dict)
        
        # Load the bounding box annotations for the image
        with open(label_path) as f:
            annotations = f.readlines()
        
        # Loop through the annotations and add them to the COCO dataset
        for ann in annotations:
            x, y, w, h = map(float, ann.strip().split()[1:])
            x_min, y_min = int((x - w / 2) * width), int((y - h / 2) * height)
            x_max, y_max = int((x + w / 2) * width), int((y + h / 2) * height)
            ann_dict = {
                "id": annos_id,
                "image_id": image_id,
                "category_id": 0,
                "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                "area": (x_max - x_min) * (y_max - y_min),
                "iscrowd": 0
            }
            coco_dataset["annotations"].append(ann_dict)
            annos_id += 1

    # Save the COCO dataset to a JSON file
    with open(os.path.join(output_path, 'annotations.json'), 'w') as f:
        json.dump(coco_dataset, f)

if __name__ == '__main__':
    Fire(main)