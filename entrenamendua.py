
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8m.pt")

# Train the model
#results = model.train(data="../data/matrikulak_entrenatzeko/data.yaml",  epochs=50, task="detect", workers=1)
#results = model.train(data="/home/ir_inf/data/matrikulak_entrenatzeko_gureak2/data.yaml",  epochs=20)
results = model.train(data="/home/ir_inf/data/License Plate.v8i.yolov8/data.yaml",  epochs=50)
