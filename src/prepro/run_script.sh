#!/bin/bash
# Базовый путь
base_path="/home/aliaksandr/Work/NkbTech/DroneHack/datasets"

# Список значений для параметра -src
base_dir=(
    "$base_path/hd/4june/train.txt"
    "$base_path/AntiUAV_add/val"
    "$base_path/AntiUAV_add/train"
    "$base_path/AntiUAV/YOLO"
    "$base_path/birds_and_drones/drone2021/yolo"
    "$base_path/birds_and_drones/sod4bird/yolo"
    "$base_path/birds_kaggle/yolo"
    "$base_path/DetFly/Det-Fly/yolo"
    "$base_path/Drones_dataset_yolo/yolo_dataset_v3_negative/train"
    "$base_path/RWODDFQTTrainDataset"
    # добавьте другие значения по мере необходимости
)

# Список значений для параметра -dst
save_path="$base_path/cropsets"
save_dir=(
    "$save_path/hd"
    "$save_path/AntiUAV_add/val"
    "$save_path/AntiUAV_add"
    "$save_path/AntiUAV"
    "$save_path/drone2021"
    "$save_path/sod4bird"
    "$save_path/birds_kaggle"
    "$save_path/Det-Fly"
    "$save_path/Drone_dataset_yolo"
    "$save_path/RWODDFQTTrainDataset"
    # добавьте другие значения по мере необходимости
)

# Список значений для параметра -log_file
log_file=(
    "$save_path/hd/big_labels.txt"
    "$save_path/AntiUAV_add/val/big_labels.txt"
    "$save_path/AntiUAV_add/big_labels.txt"
    "$save_path/AntiUAV/big_labels.txt"
    "$save_path/drone2021/big_labels.txt"
    "$save_path/sod4bird/big_labels.txt"
    "$save_path/birds_kaggle/big_labels.txt"
    "$save_path/Det-Fly/big_labels.txt"
    "$save_path/Drone_dataset_yolo/big_labels.txt"
    "$save_path/RWODDFQTTrainDataset/big_labels.txt"
)


# Убедитесь, что количество значений src и dst совпадает
if [ "${#save_dir[@]}" -ne "${#base_dir@]}" ]; then
    echo "Количество значений src и dst должно совпадать."
    exit 1
fi

# Цикл по всем значениям списка
for i in "${!save_dir[@]}"
do
    src="${base_dir[$i]}"
    dst="${save_dir[$i]}"
    log="${log_file[$i]}"
    /home/aliaksandr/Work/NkbTech/DroneHack/.conda/bin/python /home/aliaksandr/Work/NkbTech/github/dronehack24/src/prepro/smartcrop.py "$src" "$dst" --log_file "$log" 
done