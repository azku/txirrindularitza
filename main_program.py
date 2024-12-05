import urllib.request
from ultralytics import YOLO
import cv2
import os
import pandas as pd
import imutils 
from pathlib import Path

IRUDIEN_DIREKTORIOA = '../data/wpcf7-files'
IRTEERA_DIREKTORIOA = '../data/wpcf7-files/out'

# def ziurtatu_irteera_direktorioa(oraingo_direktorioa):
#     irteera_direktorioa = oraingo_direktorioa + "/out"
#     Path(oraingo_direktorioa + "/out").mkdir(parents=True, exist_ok=True)
#     return irteera_direktorioa
def argazkiari_buelta_eman(irudiaren_bide_izena):
    burua_fitxategia = os.path.split(irudiaren_bide_izena)
    helburu_bide_izena = IRTEERA_DIREKTORIOA + "/" + burua_fitxategia[1]  
    irudia = cv2.imread(irudiaren_bide_izena)
    iraulitako_irudia = imutils.rotate(irudia, angle=180)
    cv2.imwrite(helburu_bide_izena, iraulitako_irudia)
    return helburu_bide_izena

#Aurre entrenatutako modeloa
model = YOLO("yolov8m.pt")

#Emaitzak egongo diren ilara
detekzio_emaitzak = []

# Iterate through all images in the directory
for filename in os.listdir(IRUDIEN_DIREKTORIOA):
    img_path = os.path.join(IRUDIEN_DIREKTORIOA, filename)
    if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        iraulitako_irudi_bidea = argazkiari_buelta_eman(img_path)
        results = model.predict(iraulitako_irudi_bidea, verbose=False)
        detections = results[0].boxes
        for det in detections:
            if int(det.cls)  in [0, 1, 2, 3, 5, 7 ]: #[person, bicycle, car, mortorcycle, bus, truck]
                x1, y1, x2, y2 = map(int, det[0].xyxy[0])  # det.xyxy gives the box coordinates
                detekzio_emaitzak.append([iraulitako_irudi_bidea, det.conf[0].numpy(), det.cls[0].numpy(), x1, y1, x2, y2])
                detekzio_irudia = cv2.rectangle(results[0].orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.imwrite(iraulitako_irudi_bidea, detekzio_irudia)

df = pd.DataFrame(detekzio_emaitzak, columns=['image_path', 'confidence', 'type', 'bb_x1', 'bb_y1', 'bb_x2', 'bb_y2'])
df.to_csv(IRTEERA_DIREKTORIOA + "/detekzio_emaitza.csv", sep=';')
