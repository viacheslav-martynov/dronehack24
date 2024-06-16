import wandb
from ultralytics import YOLO
from wandb.integration.ultralytics import add_wandb_callback

# Step 1: Initialize a Weights & Biases run
wandb.init(project="dronehackv1_yolo8n_640_baseline_augs", job_type="training")

# Step 2: Define the YOLOv8 Model and Dataset
model_name = "yolov8n"
dataset_name = "/home/aliaksandr/Work/NkbTech/DroneHack/yaml/4june.yaml"
model = YOLO(f"{model_name}.pt")

# Step 3: Add W&B Callback for Ultralytics
add_wandb_callback(model, enable_model_checkpointing=True)

# Step 4: Train and Fine-Tune the Model
model.train(
    project="dronehackv1_yolo8n_640_baseline_augs", data=dataset_name, name = 'v8n_dronehack_4june_1280_augs_v2', device=0, 
    imgsz=1280, exist_ok=True, amp = False, batch = 15, epochs = 50,
    close_mosaic = 30  ,augment = True, lr0 =  0.0003, mosaic = 0.5)


wandb.finish()
