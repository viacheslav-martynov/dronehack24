import os
from pathlib import Path

from collections import defaultdict

import comet_ml
import json
import numpy as np

import torch
from torch.cuda.amp import GradScaler

import argparse
from tqdm import tqdm

from my_classificationlib.dataset import get_dataset
from my_classificationlib.model import get_model
from my_classificationlib.losses import get_loss
from my_classificationlib.metrics import (
    compute_metrics,
    log_metrics,
    log_confusion_matrices,
)

from my_classificationlib.utils import (
    get_experiment,
    get_optimizer,
    get_scheduler,
    log_images,
    log_grads,
    read_py_config,
)


def convert_dict_types_recursive(_dict):
    for key in _dict:
        if isinstance(_dict[key], dict):
            _dict[key] = convert_dict_types_recursive(_dict[key])
        elif isinstance(_dict[key], np.ndarray):
            _dict[key] = list(_dict[key])
    return _dict


@torch.no_grad()
def val_epoch(model, val_loader, criterion, target_names, device, cfg):
    val_confidences = defaultdict(list)
    val_predictions = defaultdict(list)
    val_ground_truth = defaultdict(list)
    val_running_loss = defaultdict(list)

    model.eval()

    batch_to_log = None
    for img, target in tqdm(val_loader, leave=False, desc="Evaluating"):
        img = img.to(device)
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=cfg.enable_mixed_presicion
        ):
            preds = model(img)
            loss = 0
            for target_name in target_names:
                target_loss = criterion(
                    preds[target_name], target[target_name].to(device)
                ).item()
                val_running_loss[target_name].append(target_loss)
                loss += target_loss

        val_running_loss["loss"].append(loss)

        for target_name in target_names:
            val_ground_truth[target_name].extend(
                target[target_name].cpu().numpy().tolist()
            )
            val_confidences[target_name].extend(
                preds[target_name]
                .softmax(dim=-1, dtype=torch.float32)
                .cpu()
                .numpy()
                .tolist()
            )
            val_predictions[target_name].extend(
                preds[target_name].argmax(dim=-1).cpu().numpy().tolist()
            )

        if batch_to_log is None:
            batch_to_log = img.to("cpu")

    results = {
        "running_loss": val_running_loss,
        "confidences": val_confidences,
        "predictions": val_predictions,
        "ground_truth": val_ground_truth,
        "images": batch_to_log,
    }

    metrics = compute_metrics(results, target_names)

    return metrics


def evaluate(model, loader, criterion, device, cfg):
    class_to_idx = loader.dataset.class_to_idx
    target_names = [*sorted(class_to_idx)]

    val_results = val_epoch(model, loader, criterion, target_names, device, cfg)

    return val_results


def main():
    parser = argparse.ArgumentParser(description="Train arguments")
    parser.add_argument(
        "-cfg", "--config", help="Config file path", type=str, default="", required=True
    )
    args = parser.parse_args()
    cfg_file = args.config
    exec(read_py_config(cfg_file), globals(), globals())
    val_loader = get_dataset(cfg.val_data, cfg.val_pipeline)
    classes = val_loader.dataset.classes
    device = torch.device(cfg.device)
    model = get_model(cfg.model, classes, device, compile=cfg.compile)
    model.set_backbone_state("freeze")
    # load weights
    model.load_state_dict(torch.load(cfg.model["checkpoint"], map_location="cpu"))
    criterion = get_loss(cfg.criterion, cfg.device)
    metrics = evaluate(model, val_loader, criterion, device, cfg)

    save_path = Path(cfg.save_path)
    save_path.mkdir(exist_ok=True, parents=True)
    with open(save_path / Path("metrics.json"), "w") as f:
        json.dump(convert_dict_types_recursive(metrics), f)


if __name__ == "__main__":
    main()
