"""
Script with code for model prediction
"""

from pathlib import Path
import os

from IPython.display import Image
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction, predict
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.utils.yolov8 import download_yolov8s_model
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.detection.iou import IntersectionOverUnion
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from fire import Fire
import time
from pathlib import Path

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

class Transforms:
    def __init__(
        self,
        transforms: A.Compose,
    ) -> None:
        self.transforms = transforms

    def __call__(
        self,
        img,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        return self.transforms(image=np.array(img))["image"]

img_size = 224
inference_pipeline = A.Compose(
    [
        A.LongestMaxSize(img_size, always_apply=True),
        A.PadIfNeeded(
            img_size,
            img_size,
            always_apply=True,
            # border_mode=cv2.BORDER_CONSTANT,
        ),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        ToTensorV2(),
    ]
)
transform = Transforms(inference_pipeline)


def get_preds_dict_from_sahi(sahi_result):
    """
    Transforms predictions from sahi output format
    to the format of torchmetrics. (pixel xyxy)
    """
    object_prediction_list = sahi_result.object_prediction_list
    boxes = [list(object_prediction.bbox.to_xyxy()) for object_prediction in object_prediction_list]
    labels = [object_prediction.category.id for object_prediction in object_prediction_list]
    scores = [object_prediction.score.value for object_prediction in object_prediction_list]
    return {
        'boxes': torch.FloatTensor(boxes),
        'scores': torch.FloatTensor(scores),
        'labels': torch.IntTensor(labels),
    }

def yolo2pixelxyxy(image_label, image_size):
    shape = [
        (image_label[0] - image_label[2] / 2) * image_size[0],
        (image_label[1] - image_label[3] / 2) * image_size[1],
        (image_label[0] + image_label[2] / 2) * image_size[0],
        (image_label[1] + image_label[3] / 2) * image_size[1]
    ]
    return shape

def get_target_dict_from_file(label_path, image_size):
    """
    Transforms yolo ground truth labels to 
    the format of torchmetrics. (pixel xyxy)
    """
    target = {
        'boxes': [],
        'labels': [],
    }
    with open(label_path, 'r') as f:
        labels = [list(map(float, label.strip().split(" "))) for label in f.readlines()]
        for label in labels:
            target['boxes'].append(yolo2pixelxyxy(label[1:], image_size))
            target['labels'].append(label[0])

    target['boxes'] = torch.FloatTensor(target['boxes'])
    target['labels'] = torch.IntTensor(target['labels'])
    return target


def get_crops_fromarray(raw_image, boxes: torch.Tensor, transform):
    """
    Crops images from original image represented as a numpy array
    Returns a torch.Tensor.
    """
    crops = []
    for box in boxes:
        crop = raw_image[int(box[1]):int(box[3]), int(box[0]):int(box[2]),:]
        crops.append(transform(crop))
    return torch.stack(crops)

def get_crops_frompil(raw_image, boxes: torch.Tensor, transform):
    """
    Crops images from original image represented as a pillow Image
    Returns a torch.Tensor.
    """
    crops = []
    for box in boxes:
        crop = raw_image.crop(list(map(int, box)))
        crops.append(transform(crop))
    return torch.stack(crops)


def predict_classifier(raw_image,
                       boxes: torch.Tensor,
                       classifier_model: torch.nn.Module,
                       transform,
                       device):
    """
    boxes : 2D tensor of shape [2, num_boxes] - pixel xyxy format
    """
    # Get crops from prediction

    if type(raw_image) is np.ndarray:
        crops = get_crops_fromarray(raw_image, boxes, transform)
    else:
        crops = get_crops_frompil(raw_image, boxes, transform)
    # predict with model
    crops = crops.to(torch.device(device))
    pred = classifier_model(crops)

    classes = (
        pred.softmax(dim=-1)
        .argmax(dim=1)
        .cpu()
        # .numpy()
        # .tolist()
    )

    return classes


def run_imgseq_validation(image_paths, labels_paths, detection_model, sahi, slice_height=640, classification_model = None):

    preds = []
    targets = []
    total_inference_time = 0
    total_boxes = 0

    for image_filename, label_filename in tqdm(zip(image_paths, labels_paths), total=len(image_paths)):
        image = Image.open(image_filename)
        start_time = time.time()
        
        if sahi:
            result_sahi = get_sliced_prediction(
                image,
                detection_model,
                slice_height=slice_height,
                slice_width=slice_height,

                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,
                verbose=False,
            )
        else:
            result_sahi = get_prediction(
                image,
                detection_model,
                verbose=False,
            )
        
        inference_time = time.time() - start_time
        total_inference_time += inference_time

        pred = get_preds_dict_from_sahi(result_sahi)

        ### Optional stage to predict class with a classifier
        if classification_model:
            pred['labels'] = predict_classifier(
                image,
                pred['boxes'],
                classification_model,
                transform,
                device='cuda')

        preds.append(pred)
        targets.append(get_target_dict_from_file(label_filename, image.size))

        total_boxes += len(pred['boxes'])

    metrics = MeanAveragePrecision(box_format='xywh', iou_type='bbox')
    metrics_dict = metrics(preds, targets)
    
    average_inference_time = total_inference_time / len(image_paths)
    average_boxes_per_image = total_boxes / len(image_paths)

    metrics_dict['average_inference_time'] = average_inference_time
    metrics_dict['average_boxes_per_image'] = average_boxes_per_image

    return metrics_dict

def process_weights_directory(weights_directory: Path, image_paths, labels_paths, device, confidence_threshold, save_path, sahi, slice_height):
    weights_path = weights_directory / 'weights' / 'best.pt'
    if not weights_path.exists():
        print(f"Skipping {weights_directory} as no best.pt found in weights directory.")
        return

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=str(weights_path),
        confidence_threshold=confidence_threshold,
        device=device
    )

    metrics = run_imgseq_validation(image_paths, labels_paths, detection_model, sahi, slice_height)
    metrics.pop('classes', None)

    for key in metrics:
        try:
            metrics[key] = float(metrics[key])
        except:
            pass

    # Select only the 1st, 2nd, and 3rd columns, plus the added columns
    selected_metrics = {key: metrics[key] for key in list(metrics)[:3]}
    selected_metrics['average_inference_time'] = metrics['average_inference_time']
    selected_metrics['average_boxes_per_image'] = metrics['average_boxes_per_image']

    metrics_df = pd.DataFrame(selected_metrics, index=['value'])
    save_path = Path(save_path)
    # Construct the filename based on the weights path and sahi parameter
    base_name = Path(weights_path).parent.parent.stem
    sahi_suffix = f"+sahi{slice_height}" if sahi else "+nosahi"
    foldername = f"{base_name}{sahi_suffix}"

    i = 0
    while os.path.exists(Path(save_path, f'{foldername}_{i}')):
        i += 1
     
    os.makedirs(Path(save_path, f'{foldername}_{i}'))

    metrics_df.to_csv(Path(save_path, f'{foldername}_{i}',f'metrics.csv'))

def process_weights(
        weights_detector: Path,
        image_paths,
        labels_paths,
        device,
        confidence_threshold,
        save_path,
        sahi,
        slice_height,
        weights_classifier: None):

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=str(weights_detector),
        confidence_threshold=confidence_threshold,
        device=device
    )

    if weights_classifier:
        classifier_model = torch.load(weights_classifier, map_location=torch.device(device))
    else:
        classifier_model = None

    metrics = run_imgseq_validation(
        image_paths,
        labels_paths,
        detection_model,
        sahi,
        slice_height,
        classification_model=classifier_model)
    metrics.pop('classes', None)

    for key in metrics:
        try:
            metrics[key] = float(metrics[key])
        except:
            pass

    # Select only the 1st, 2nd, and 3rd columns, plus the added columns
    selected_metrics = {key: metrics[key] for key in list(metrics)[:3]}
    selected_metrics['average_inference_time'] = metrics['average_inference_time']
    selected_metrics['average_boxes_per_image'] = metrics['average_boxes_per_image']

    metrics_df = pd.DataFrame(selected_metrics, index=['value'])
    save_path = Path(save_path)
    # Construct the filename based on the weights path and sahi parameter
    base_name = weights_detector.stem
    sahi_suffix = f"+sahi{slice_height}" if sahi else "+nosahi"
    foldername = f"{base_name}{sahi_suffix}"

    i = 0
    while os.path.exists(Path(save_path, f'{foldername}_{i}')):
        i += 1
     
    os.makedirs(Path(save_path, f'{foldername}_{i}'))

    metrics_df.to_csv(Path(save_path, f'{foldername}_{i}',f'metrics.csv'))


def main(image_paths_filename: str,
         detector_weights_path: str,
         classifier_weights_path: str = None,
         device='cuda:0',
         confidence_threshold=0.25, 
         save_path='runs',
         sahi: bool = True,
         slice_height=640
    ):
    with open(image_paths_filename, 'r') as f:
        image_paths = [Path(filename.strip()) for filename in f.readlines()]
    labels_paths = [Path(*image_path.parts[:-2], 'labels', image_path.parts[-1]).with_suffix('.txt') for image_path in image_paths]

    detector_weights_path = Path(detector_weights_path)
    classifier_weights_path = Path(classifier_weights_path)

    if detector_weights_path.is_dir():
        for subdir in detector_weights_path.iterdir():
            if subdir.is_dir():
                process_weights_directory(subdir, image_paths, labels_paths, device, confidence_threshold, save_path, sahi, slice_height)
    

    else:
        process_weights(
            detector_weights_path,
            image_paths,
            labels_paths,
            device,
            confidence_threshold,
            save_path,
            sahi,
            slice_height,
            weights_classifier = classifier_weights_path,
        )

# def main(image_paths_filename: str, weights_path: str, device='cuda:0', confidence_threshold=0.25, save_path='runs', sahi: bool = True, slice_height=640):
#     with open(image_paths_filename, 'r') as f:
#         image_paths = [Path(filename.strip()) for filename in f.readlines()]
#     labels_paths = [Path(*image_path.parts[:-2], 'labels', image_path.parts[-1]).with_suffix('.txt') for image_path in image_paths]

#     weights_path = Path(weights_path)
#     run_imgseq_validation(image_paths, labels_paths, detection_model, sahi, slice_height=640):

if __name__ == "__main__":
    Fire(main)