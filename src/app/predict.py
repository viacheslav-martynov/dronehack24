"""
Models for inference
"""
import os
import cv2
from datetime import datetime, timedelta
import numpy as np
import torch
import shutil

# # from ultralytics import YOLO, YOLOv10
from sahi import AutoDetectionModel
from PIL import Image
from sahi.predict import get_prediction, get_sliced_prediction, predict
from results import Results
from pathlib import Path
# from ultralytics.engine.results import Results


def get_preds_dict_from_sahi(sahi_result):
    """
    Accepts sahi prediction result and returns dict of boxes, scores, and labels
    boxes are returned in pixel xyxy format !
    label indices are returned !
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

def get_yolo_result(result_sahi, category_mapping):
    if len(result_sahi.object_prediction_list) > 0:
        bboxes = torch.stack(
            [
                torch.Tensor(object_prediction.bbox.to_xyxy()) for object_prediction in result_sahi.object_prediction_list]
        )
        scores = torch.Tensor(
            [object_prediction.score.value for object_prediction in result_sahi.object_prediction_list]
        )
        categories = torch.Tensor(
            [object_prediction.category.id for object_prediction in result_sahi.object_prediction_list]
        )
        try:
            result_tensor = torch.cat([
                bboxes,
                scores.unsqueeze(0).T,
                categories.unsqueeze(0).T], axis=1)
        except:
            import ipdb; ipdb.set_trace()
    
    else:
        result_tensor = None

    yolo_result = Results(
        np.array(result_sahi.image),
        path=None,
        names=category_mapping,
        boxes=result_tensor)
    return yolo_result


def convert_category_keys(category_mapping):
    return dict(
                zip(
                    list(map(int, category_mapping.keys())),
                    category_mapping.values(),
                )
            )

def update_timeline_data(result_sahi, timeline_data: dict):
    categories = set([object_prediction.category.id for object_prediction in result_sahi.object_prediction_list])
    for category in timeline_data.keys():
        if category in categories:
            timeline_data[category].append(1)
        else:
            timeline_data[category].append(0)
    return timeline_data

def finalize_timeline_data(timeline_data: list, fps: float):
    # TODO check how it works with the last element = 0 ([1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,0,1,0,1,0])
    for category in timeline_data.keys():
        index = 0
        while index<len(timeline_data[category]):
            # if detection is found -> fill one second forward and backward
            if timeline_data[category][index] == 1:
                assignment_len = int(np.clip(index+fps, a_min = 0, a_max=len(timeline_data[category]))) - int(np.clip(index-fps, a_min=0, a_max=len(timeline_data[category])))
                # set these entries with one
                timeline_data[category][
                    int(np.clip(index-fps, a_min = 0, a_max=len(timeline_data[category]))):int(np.clip(index+fps, a_min = 0, a_max = len(timeline_data[category])))
                    ] = [1 for _ in range(assignment_len)]
                index=int(np.clip(index+fps, a_min=0, a_max=len(timeline_data[category]))-1)
            index+=1

    # 2. detect 0-1, 1-0 value changes and fill corresponding timecodes
    # here we imagine that video starts at current time 
    # each detection is added in timeline at current time + detection time
    items = []
    n_item = 0
    now = datetime.now()
    
    for category in timeline_data.keys():
        prev_value = 0
        item = {}
        for index, current_value in enumerate(timeline_data[category]):
            # detect value change from 0 to 1
            if current_value - prev_value > 0:
                item['id'] = n_item
                item['content'] = str(category)
                if (item['content'] == '4') or (item['content'] == '0'):
                    item['style'] = "background-color: red"
                start_sec = int(index/fps)
                start_time = now + timedelta(seconds=start_sec)
                item['start'] = start_time.strftime("%Y/%m/%d %H:%M:%S")
            # detect value change from 1 to 0
            if current_value - prev_value < 0:
                end_sec = int(index/fps)
                end_time = now + timedelta(seconds=end_sec)
                item['end'] = end_time.strftime("%Y/%m/%d %H:%M:%S")

            # conditions to add item to list
            if ('start' in item.keys()) and ('end' in item.keys()):
                items.append(item)
                item = {}
                n_item+=1
            prev_value = current_value

        #  if after iteration we have an item with 'start' key but no 'end' key:
        if ('start' in item.keys()) and ('end' not in item.keys()):
            end_sec = int(index/fps)
            end_time = now + timedelta(seconds=end_sec)
            item['end'] = end_time.strftime("%Y/%m/%d %H:%M:%S")
            items.append(item)
            n_item+=1


    return items, now

def predict_frame(detection_model, image: Image, slice_height=1280, slice_width=1280):
    raise RuntimeError('Dont use this function, check category mapping keys type')
    result_sahi = get_sliced_prediction(
        image,
        detection_model,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        verbose=False,
    )
    return get_yolo_result(result_sahi, detection_model.category_mapping)

def predict_image_directory(
        detection_model,
        image_dir,
        sahi=False,
        should_save_preds=True,
        should_save_visualisation=False,
        save_path='.',
        slice_height=1280, slice_width=1280):
    
    shutil.rmtree(save_path, ignore_errors=True)
    os.makedirs(save_path, exist_ok=True)
    
    category_ids = list(map(int, list(detection_model.category_mapping.keys())))
    predictions = []
    for path in Path(image_dir).iterdir():
        if path.is_file():
            image = Image.open(path)
            if sahi:
                result_sahi = get_sliced_prediction(
                    image,
                    detection_model,
                    slice_height=slice_height,
                    slice_width=slice_width,
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
            result_yolo = get_yolo_result(result_sahi, convert_category_keys(detection_model.category_mapping))
            if should_save_preds:
                result_yolo.save_txt(Path(save_path, path.stem).with_suffix('.txt'))

def predict_video(
        detection_model,
        video_path,
        sahi=False,
        should_save_preds=False,
        should_save_video=True,
        save_filename='pred.mp4',
        slice_height=1280, slice_width=1280
    ):
    
    # TODO нужно ли хранить предсказания с кадров??? они занимают место в памяти ;(
    # по факту, сценарий использования метода - вызвал, сохранил видео, предикты с видео - не нужны.
    category_ids = list(map(int, list(detection_model.category_mapping.keys())))
    
    vidcap = cv2.VideoCapture(video_path)

    fps = vidcap.get(cv2.CAP_PROP_FPS)
    width  = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    if should_save_video:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        output = cv2.VideoWriter(save_filename, fourcc, fps,(int(width),int(height)))

    pred_timeline = {category_id: [] for category_id in category_ids}
    results_yolo = []

    ret, image = vidcap.read()

    if sahi:
        result_sahi = get_sliced_prediction(
            image,
            detection_model,
            slice_height=slice_height,
            slice_width=slice_width,
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

    result_yolo = get_yolo_result(result_sahi, convert_category_keys(detection_model.category_mapping))
    
    # update prediction timeline
    pred_timeline = update_timeline_data(result_sahi, pred_timeline)

    if should_save_preds:
        results_yolo.append(result_yolo)

    if should_save_video:
        output.write(result_yolo.plot())

    while ret:
        ret, image = vidcap.read()
        if ret:
            if sahi:
                result_sahi = get_sliced_prediction(
                    image,
                    detection_model,
                    slice_height=slice_height,
                    slice_width=slice_width,
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
            pred_timeline = update_timeline_data(result_sahi, pred_timeline)
            
            result_yolo = get_yolo_result(result_sahi, convert_category_keys(detection_model.category_mapping))
            if should_save_preds:
                results_yolo.append(result_yolo)
                
            if should_save_video:
                output.write(result_yolo.plot())
    output.release()  
    vidcap.release()

    timeline_items, start_timestamp = finalize_timeline_data(pred_timeline, fps)

    return results_yolo, timeline_items, start_timestamp



