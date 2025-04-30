# train.py

import os
from yolov5 import train  # This imports YOLOv5's training function

# Set paths
DATA_YAML = 'data.yaml'  # Path to your data.yaml
WEIGHTS = 'yolov5s.pt'   # Start training from pre-trained yolov5s weights (small model)
IMG_SIZE = 640           # Image size
BATCH_SIZE = 16          # Batch size
EPOCHS = 100              # Number of epochs
PROJECT = 'cacao_training'  # Folder where results will be saved

# Make sure YOLOv5 repo is installed
if not os.path.exists('yolov5'):
    os.system('git clone https://github.com/ultralytics/yolov5.git')
    os.system('pip install -r yolov5/requirements.txt')

# Run training
train.run(
    data=DATA_YAML,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    epochs=EPOCHS,
    weights=WEIGHTS,
    project=PROJECT,
    name='exp',
    exist_ok=True  # Overwrite existing
)
