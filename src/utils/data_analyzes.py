import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import numpy as np
from tqdm import tqdm




# Определение классов
class_labels = {
    0: 'copter_type_uav',
    1: 'aircraft',
    2: 'helicopter',
    3: 'bird',
    4: 'aircraft_type_uav'
}

def load_annotations(image_folders):
    data = []
    for image_folder in image_folders:
        label_folder = image_folder.replace('images', 'labels')
        annotation_files = glob.glob(os.path.join(label_folder, '*.txt'))

        for file in tqdm(annotation_files, desc=f"Loading annotations from {label_folder}"):
            with open(file, 'r') as f:
                for line in f.readlines():
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    image_name = os.path.basename(file).replace('.txt', '.jpg')
                    data.append([image_name, class_id, x_center, y_center, width, height])
    
    df = pd.DataFrame(data, columns=['image_name', 'class_id', 'x_center', 'y_center', 'width', 'height'])
    df['class_name'] = df['class_id'].apply(lambda x: class_labels[int(x)])
    return df

def plot_class_distribution(df, ax):
    sns.countplot(x='class_name', data=df, ax=ax)
    ax.set_title('Class Distribution')
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

def plot_relative_area_distribution(df, ax):
    df['area'] = df['width'] * df['height']
    sns.boxplot(x='class_name', y='area', data=df, ax=ax)
    ax.set_title('Relative Area Distribution by Class')
    ax.set_xlabel('Class')
    ax.set_ylabel('Relative Area')
    ax.set_yscale('log')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

def plot_heatmap(df, image_folders, ax):
    image_sizes = {}
    for image_folder in image_folders:
        image_files = glob.glob(os.path.join(image_folder, '*.jpg'))
        for file in tqdm(image_files, desc=f"Loading image sizes from {image_folder}"):
            try:
                image_sizes[os.path.basename(file)] = plt.imread(file).shape[:2]
            except OSError as e:
                print(f"Warning: {file} could not be loaded. {e}")
    
    heatmap = np.zeros((1000, 1000))
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Creating heatmap"):
        image_name = row['image_name']
        if image_name not in image_sizes:
            continue
        img_height, img_width = image_sizes[image_name]
        x_center = int(row['x_center'] * 1000)
        y_center = int(row['y_center'] * 1000)
        width = int(row['width'] * 1000)
        height = int(row['height'] * 1000)
        x1, x2 = max(0, x_center - width // 2), min(1000, x_center + width // 2)
        y1, y2 = max(0, y_center - height // 2), min(1000, y_center + height // 2)
        heatmap[y1:y2, x1:x2] += 1
    
    sns.heatmap(heatmap, cmap='hot', cbar=True, ax=ax)
    ax.set_title('Bounding Box Heatmap')

def plot_image_area_distribution(image_folders, ax):
    areas = []
    sizes = set()
    for image_folder in image_folders:
        image_files = glob.glob(os.path.join(image_folder, '*.jpg'))
        for file in tqdm(image_files, desc=f"Getting image sizes from {image_folder}"):
            try:
                height, width = plt.imread(file).shape[:2]
                areas.append(height * width)
                sizes.add((height, width))
            except OSError as e:
                print(f"Warning: {file} could not be loaded. {e}")

    if len(sizes) > 1:
        df_areas = pd.DataFrame(areas, columns=['Area'])
        sns.histplot(df_areas['Area'], kde=True, color='green', ax=ax)
        ax.set_title('Image Area Distribution')
        ax.set_xlabel('Area (pixels²)')
        ax.set_ylabel('Frequency')
        return True
    else:
        ax.set_visible(False)
        return False

def plot_bounding_boxes(df, ax, sample_size=1000):
    if len(df) > sample_size:
        df = df.sample(sample_size)
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Plotting bounding boxes"):
        x_center, y_center, width, height = row['x_center'], row['y_center'], row['width'], row['height']
        x1, x2 = x_center - width / 2, x_center + width / 2
        y1, y2 = y_center - height / 2, y_center + height / 2
        rect = plt.Rectangle((x1, y1), width, height, linewidth=0.5, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Bounding Boxes')
    ax.set_aspect('equal')


    # Usage
image_folders = ['/home/aliaksandr/Work/NkbTech/DroneHack/datasets/RWODDFQTTrainDataset/images']  # Укажите ваши папки здесь
df = load_annotations(image_folders)

fig, axes = plt.subplots(3, 2, figsize=(15, 18))
axes = axes.flatten()

plot_class_distribution(df, axes[0])
plot_relative_area_distribution(df, axes[1])
plot_heatmap(df, image_folders, axes[2])
plot_bounding_boxes(df, axes[3])
# Условно строим график распределения площади изображений
image_area_plot_successful = plot_image_area_distribution(image_folders, axes[4])

# Если распределение площади изображений не построено, строим bounding boxes на его месте
if not image_area_plot_successful:
    fig.delaxes(axes[4])  # Удаляем последнюю пустую ось

# Оставляем последний график пустым или можно добавить ещё один график, если требуется
fig.delaxes(axes[5])

plt.tight_layout()

# Определение пути для сохранения изображения
save_path = os.path.join(os.path.dirname(image_folders[0]), 'all_plots.png')
plt.savefig(save_path)
print(f"All plots saved to {save_path}")
