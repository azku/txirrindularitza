import urllib.request
from ultralytics import YOLO
import cv2
import os
import pandas as pd


#Model
model = YOLO("yolov8m.pt")

# Directory containing images
#image_dir = 'data'
image_dir = '../mini_data'

#Emaitzak egongo diren ilara
rows = []

# Iterate through all images in the directory
for filename in os.listdir(image_dir):
    img_path = os.path.join(image_dir, filename)
    if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        results = model.predict(img_path, verbose=False)
        detections = results[0].boxes
        for det in detections:
            if int(det.cls)  in [0, 1, 2, 3, 5, 7 ]: #[person, bicycle, car, mortorcycle, bus, truck]
                x1, y1, x2, y2 = map(int, det[0].xyxy[0])  # det.xyxy gives the box coordinates
                rows.append([img_path, det.conf[0].numpy(), det.cls[0].numpy(), x1, y1, x2, y2])
                
                #cv2.rectangle(results[0].orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

df = pd.DataFrame(rows, columns=['image_path', 'confidence', 'type', 'bb_x1', 'bb_y1', 'bb_x2', 'bb_y2'])
df.to_csv("detekzio_emaitza.csv", sep=';')
