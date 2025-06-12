# Fine-tune YOLOv11n on your custom dataset using Ultralytics CLI
# Run this script in your project root (where this script is located)

import os

# Paths (update if your structure changes)
data_yaml = "robot-uas/dataset-train/person detection/data.yaml"
pretrained_model = "yolov11n"  # Make sure this file is present or will be downloaded by Ultralytics
epochs = 50
imgsz = 640

# Command to run
command = f"yolo task=detect mode=train model={pretrained_model} data=\"{data_yaml}\" epochs={epochs} imgsz={imgsz}"

print("Running YOLOv11n fine-tuning with the following command:")
print(command)
os.system(command)
print("\nTraining complete. Check runs/detect/train/weights/best.pt for your fine-tuned model.")
