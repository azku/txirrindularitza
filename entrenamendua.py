
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8m.pt")

# Train the model
results = model.train(data="../data/matrikulak_entrenatzeko/data.yaml",  epochs=50, task="detect", workers=1)
