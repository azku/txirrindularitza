import urllib.request
from ultralytics import YOLO
import cv2
import os
import pandas as pd
import imutils 
from pathlib import Path
import numpy as np
import argazkien_metadatoak_ekarri as arg_meta

IRUDIEN_DIREKTORIOA = '../data/wpcf7-files'
IRTEERA_DIREKTORIOA = '../data/wpcf7-files/out'

# def ziurtatu_irteera_direktorioa(oraingo_direktorioa):
#     irteera_direktorioa = oraingo_direktorioa + "/out"
#     Path(oraingo_direktorioa + "/out").mkdir(parents=True, exist_ok=True)
#     return irteera_direktorioa

def direktorioko_irudiak_igaro_eraldatu(sarrera_direktorioa,irteera_izena, albo_ondorioa):
    for uneko_fitxategia in os.listdir(sarrera_direktorioa):
        img_path = os.path.join(sarrera_direktorioa, uneko_fitxategia)
        irteera_direktorioa = sarrera_direktorioa + "/" + irteera_izena
        #Ziurtatu existitzen dala
        Path(irteera_direktorioa).mkdir(parents=True, exist_ok=True)
        if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            albo_ondorioa(uneko_fitxategia, irteera_direktorioa)

def argazkiei_buelta_eman(sarrera_direktiorioa, irteera_izena):
    direktorioko_irudiak_igaro_eraldatu(sarrera_direktiorioa, irteera_izena, argazkiari_buelta_eman)
def argazkiari_buelta_eman(uneko_fitxategia, irteera_direktorioa):
    print("Buelta ematen")
    
def argazkiari_buelta_eman(irudiaren_bide_izena):
    burua_fitxategia = os.path.split(irudiaren_bide_izena)
    helburu_bide_izena = IRTEERA_DIREKTORIOA + "/" + burua_fitxategia[1]  
    irudia = cv2.imread(irudiaren_bide_izena, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
    iraulitako_irudia =    imutils.rotate(irudia, angle=180)
    cv2.imwrite(helburu_bide_izena, iraulitako_irudia)
    return helburu_bide_izena

#Aurre entrenatutako modeloa
model = YOLO("yolov8m.pt")
matrikula_modeloa = YOLO("license_plate_detector.pt")

#Emaitzak egongo diren ilara
detekzio_emaitzak = []

def identifikatu_izeneko_datuak(filename):
    s = filename.split('_')
    return (s[1],s[2],s[3]) #miliseconds, random, distance

def matrikula_detekzioa(detekzio_irudia):
    m_results = matrikula_modeloa.predict(detekzio_irudia, verbose=False)
    m_detections = m_results[0].boxes
    for det in m_detections:
        x1, y1, x2, y2 = map(int, det[0].xyxy[0])
        if  det.conf[0]>0.3:
            cv2.rectangle(detekzio_irudia, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(detekzio_irudia, f"{m_results[0].names[int(det.cls)]} ({ det.conf[0].numpy()})", (x1, y1 +20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
arg_hiztegia = arg_meta.get_picture_metadata()

# Iterate through all images in the directory
for filename in os.listdir(IRUDIEN_DIREKTORIOA):
    img_path = os.path.join(IRUDIEN_DIREKTORIOA, filename)
    if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        ml, rd, dist = identifikatu_izeneko_datuak(filename)
        iraulitako_irudi_bidea = argazkiari_buelta_eman(img_path)
        results = model.predict(iraulitako_irudi_bidea, verbose=False)
        detections = results[0].boxes
        for det in detections:
            if int(det.cls)  in [0, 1, 2, 3, 5, 7 ]: #[person, bicycle, car, mortorcycle, bus, truck]
                x1, y1, x2, y2 = map(int, det[0].xyxy[0])  # det.xyxy gives the box coordinates
                bb_azalera = (x2 - x1) * (y2 - y1)
                metadatuak = arg_hiztegia[filename]
                detekzio_emaitzak.append([metadatuak["id"], metadatuak["date"], metadatuak["time"], metadatuak["route"], ml, dist, iraulitako_irudi_bidea, det.conf[0].numpy(), det.cls[0].numpy(), x1, y1, x2, y2, bb_azalera])
                detekzio_irudia = None
                if bb_azalera > 200000:
                    #bounding box bigenough. If object is car den size should be bigger than...
                    detekzio_irudia = cv2.rectangle(results[0].orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(detekzio_irudia, results[0].names[int(det.cls)], (x1, y1 +20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    detekzio_irudia = cv2.rectangle(results[0].orig_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    #cv2.putText(detekzio_irudia, results[0].names[int(det.cls)], (x1, y1 +20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                if int(det.cls)  in [2, 3, 5, 7 ]: #[car, mortorcycle, bus, truck]
                    #matrikula detekzioa
                    matrikula_detekzioa(detekzio_irudia)
                cv2.imwrite(iraulitako_irudi_bidea, detekzio_irudia)
                

df = pd.DataFrame(detekzio_emaitzak, columns=['id', 'date', 'time', 'route', 'miliseconds', 'distance', 'image_path', 'confidence', 'type', 'bb_x1', 'bb_y1', 'bb_x2', 'bb_y2', 'area'])
df.to_csv(IRTEERA_DIREKTORIOA + "/detekzio_emaitza.csv", sep=';')
