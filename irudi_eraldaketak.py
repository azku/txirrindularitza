import os
import imutils 
import cv2
from pathlib import Path
from itertools import chain
from ultralytics import YOLO
import numpy
import argazkien_metadatoak_ekarri as arg_meta
import pandas as pd

def get_direktorioko_irudiak(sarrera_direktorioa):
    irudien_lista = filter(lambda x: os.path.join(sarrera_direktorioa, x).lower().endswith(('.png', '.jpg', '.jpeg')),
           os.listdir(sarrera_direktorioa))
    return map(lambda x: os.path.join(sarrera_direktorioa, x), irudien_lista)

def direktorioko_irudiak_igaro_eraldatu(sarrera_direktorioa,irteera_izena, albo_ondorioa):
    for uneko_fitxategia in os.listdir(sarrera_direktorioa):
        fitxategi_helbide_osoa = os.path.join(sarrera_direktorioa, uneko_fitxategia)
        irteera_direktorioa = sarrera_direktorioa + "/" + irteera_izena
        #Ziurtatu existitzen dala
        Path(irteera_direktorioa).mkdir(parents=True, exist_ok=True)
        if fitxategi_helbide_osoa.lower().endswith(('.png', '.jpg', '.jpeg')):
            albo_ondorioa(fitxategi_helbide_osoa, irteera_direktorioa)

def argazkiei_buelta_eman(sarrera_direktorioa, irteera_izena, anguloa=180):
    irteera_direktorioa = sarrera_direktorioa + "/" + irteera_izena
    Path(irteera_direktorioa).mkdir(parents=True, exist_ok=True)
    for fitxategi_helbide_osoa in get_direktorioko_irudiak(sarrera_direktorioa):
        argazkiari_buelta_eman(fitxategi_helbide_osoa,  irteera_direktorioa, anguloa)

def argazkiari_buelta_eman(fitxategi_helbide_osoa, irteera_direktorioa, anguloa=180):
    burua_fitxategia = os.path.split(fitxategi_helbide_osoa)
    helburu_bide_izena = irteera_direktorioa + "/" + burua_fitxategia[1]
    irudia = cv2.imread(fitxategi_helbide_osoa, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
    iraulitako_irudia =    imutils.rotate(irudia, angle=anguloa)
    cv2.imwrite(helburu_bide_izena, iraulitako_irudia)

def predikzioak_burutu(sarrera_direktorioa, modeloaren_izena, mozketa_azalera=200000):
    model = YOLO(modeloaren_izena)
    def predikzioa_burutu(fitxategi_helbide_osoa):
        predikzio_emaitza = model.predict(fitxategi_helbide_osoa, verbose=False)[0]
        for det in  predikzio_emaitza.boxes:
            #if int(det.cls)  in [0, 1, 2, 3, 5, 7 ]:
            x1, y1, x2, y2 = map(int, det[0].xyxy[0])  # det.xyxy gives the box coordinates
            bb_azalera = (x2 - x1) * (y2 - y1)
            yield {"fitxategi_helbide_osoa":fitxategi_helbide_osoa,
                   "irudi_originala": predikzio_emaitza.orig_img,
                   "konfiantza": float(det.conf),
                   "etiketa":  predikzio_emaitza.names[int(det.cls)],
                   "x1": x1, "y1":y1, "x2": x2, "y2":y2,
                   "bb_azalera": bb_azalera}
    return chain.from_iterable(map(predikzioa_burutu, get_direktorioko_irudiak(sarrera_direktorioa)))

def predikzioak_burutu_obb(sarrera_direktorioa, modeloaren_izena, mozketa_azalera=200000):
    model = YOLO(modeloaren_izena)
    def predikzioa_burutu(fitxategi_helbide_osoa):
        predikzio_emaitza = model.predict(fitxategi_helbide_osoa, verbose=False)[0]
        for det in  predikzio_emaitza.obb:
            #if int(det.cls)  in [0, 1, 2, 3, 5, 7 ]:
            #a1, a1, a3, a4 =  det[0].xyxyxyxy[0].detach().numpy()
            #bb_azalera = (x2 - x1) * (y2 - y1)
            yield {"fitxategi_helbide_osoa": fitxategi_helbide_osoa,
                   "irudi_originala": predikzio_emaitza.orig_img,
                   "konfiantza": float(det.conf),
                   "etiketa":  predikzio_emaitza.names[int(det.cls)],
                   "coord": det.xyxyxyxy[0].detach().numpy(),
                   "bb_azalera": 0}
    return chain.from_iterable(map(predikzioa_burutu, get_direktorioko_irudiak(sarrera_direktorioa)))

def draw_label_and_confidence_4points(image, label, confidence, points):
    """
    Draws a label and confidence score on the image near a quadrilateral defined by 4 points.
    
    :param image: The input image
    :param label: The label for the object (e.g., 'Dog')
    :param confidence: The confidence score (e.g., 0.89)
    :param points: A list of 4 points [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] defining the bounding box
    :return: The image with the label and confidence drawn
    """
    # Unpack the points (top-left, top-right, bottom-right, bottom-left)
    top_left, top_right, bottom_right, bottom_left = points
    
    # Calculate the top-left corner and width/height for drawing the text
    x1, y1 = top_left
    x2, y2 = bottom_right
    
    # Prepare the label and confidence text
    label_text = f"{label}: {confidence:.2f}"

    # Set the font and text color
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (0, 255, 0)  # Green color for text
    thickness = 1

    # Calculate the position for the label text
    (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
    
    # Place the text above the bounding box or within the image bounds
    text_x = x1
    text_y = y1 - 5  # Position the text slightly above the bounding box

    # If the label is too long and goes beyond the image boundary, adjust position
    if text_x + text_width > image.shape[1]:
        text_x = image.shape[1] - text_width - 5

    # Draw the polygon (bounding box) using the 4 points
    points_array = numpy.array([top_left, top_right, bottom_right, bottom_left], numpy.int32)
    points_array = points_array.reshape((-1, 1, 2))
    cv2.polylines(image, [points_array], isClosed=True, color=(0, 255, 0), thickness=2)
    
    # Draw the label and confidence
    cv2.putText(image, label_text, (text_x, text_y), font, font_scale, color, thickness, lineType=cv2.LINE_AA)

    return image

def predikzioa_margoztu_obb(irudi_originala, coord, etiketa, konfiantza, irteera_helbidea):
    # detekzio_irudia = cv2.polylines(predikzio_emaitza.orig_img, [ct], 1, (0, 255, 0))
    detekzio_irudia = cv2.drawContours(irudi_originala, [coord.astype(numpy.int32)], 0, (0, 255, 0), 2)
    detekzio_irudia = draw_label_and_confidence_4points(detekzio_irudia, etiketa, konfiantza, coord.astype(numpy.int32))
    #cv2.putText(detekzio_irudia, etiketa, (x1, y1 +20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    #cv2.putText(detekzio_irudia, str(round(konfiantza,2)), (x1, y1 +50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imwrite(irteera_helbidea, detekzio_irudia)
def predikzioak_margoztu_obb(predikzio_hiztegia, sarrera_direktorio_helbidea, irteera_direktorio_izena, konfiantza_minimoa=0.4):
    helburu_direktorio_helbidea = sarrera_direktorio_helbidea + "/" + irteera_direktorio_izena
    Path(helburu_direktorio_helbidea).mkdir(parents=True, exist_ok=True)
    for p in predikzio_hiztegia:
        if p["konfiantza"]>0.4:
            helbide_zatitua = os.path.split(p["fitxategi_helbide_osoa"])
            helburu_bide_izena = helburu_direktorio_helbidea + "/" + helbide_zatitua[1]
            predikzioa_margoztu_obb(p['irudi_originala'], p["coord"], p["etiketa"], p["konfiantza"], helburu_bide_izena)    

def predikzioa_margoztu(irudi_originala, x1, y1, x2, y2, etiketa, konfiantza, irteera_helbidea):
    detekzio_irudia = cv2.rectangle(irudi_originala, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(detekzio_irudia, etiketa, (x1, y1 +20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(detekzio_irudia, str(round(konfiantza,2)), (x1, y1 +50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imwrite(irteera_helbidea, detekzio_irudia)

def predikzioak_margoztu(predikzio_hiztegia, sarrera_direktorio_helbidea, irteera_direktorio_izena, konfiantza_minimoa=0.4):
    helburu_direktorio_helbidea = sarrera_direktorio_helbidea + "/" + irteera_direktorio_izena
    Path(helburu_direktorio_helbidea).mkdir(parents=True, exist_ok=True)
    for p in predikzio_hiztegia:
        if p["konfiantza"]>0.4:
            helbide_zatitua = os.path.split(p["fitxategi_helbide_osoa"])
            helburu_bide_izena = helburu_direktorio_helbidea + "/" + helbide_zatitua[1]
            predikzioa_margoztu(p['irudi_originala'], p["x1"], p["y1"], p["x2"], p["y2"], p["etiketa"], p["konfiantza"], helburu_bide_izena)

def identifikatu_izeneko_datuak(filename):
    s = filename.split('_')
    return (s[1],s[2],s[3]) #miliseconds, random, distance

def predikzioak_dataframe_bihurtu(predikzio_hiztegia):
    metadatu_hiztegia = arg_meta.get_picture_metadata()
    df_lista = []
    for p in predikzio_hiztegia:
        helbide_zatitua = os.path.split(p["fitxategi_helbide_osoa"])
        ml, rd, dist = identifikatu_izeneko_datuak(helbide_zatitua[1])
        if helbide_zatitua[1] in metadatu_hiztegia:
            metadatuak = metadatu_hiztegia[helbide_zatitua[1]]
        else:
            print(f"EZ da aurkitu {helbide_zatitua}")
            metadatuak = {"id": None, "date": None, "time": None, "route": None}
        # df_lista.append({"id": metadatuak["id"], "date": metadatuak["date"], "time": metadatuak["time"], "route": metadatuak["route"],
        #                  'miliseconds': ml, 'distance': dist, 'image_path': p["fitxategi_helbide_osoa"],
        #                  'confidence':str(p["konfiantza"]) , 'type': p["etiketa"], 'bb_x1': p["x1"], 'bb_y1': p["x2"],
        #                  'bb_x2': p["x2"], 'bb_y2': p["y2"], 'area': (p["x2"] - p["x1"]) * (p["y2"] - p["y1"])})
        df_lista.append({"id": metadatuak["id"], "date": metadatuak["date"], "time": metadatuak["time"], "route": metadatuak["route"],
                         'miliseconds': ml, 'distance': dist, 'image_path': p["fitxategi_helbide_osoa"],
                         'confidence':str(p["konfiantza"]) , 'type': p["etiketa"], 'bb_x1': p["coord"]})
    return pd.DataFrame(df_lista)
        
# def detekzioak_margoztu(predikzio_emaitza, irudia_helbide_osoa, irteera_helbide_osoa, mozketa_azalera=200000):
#     for det in predikzio_emaitza.boxes:
#         if int(det.cls)  in [0, 1, 2, 3, 5, 7 ]: #[person, bicycle, car, mortorcycle, bus, truck]
#             x1, y1, x2, y2 = map(int, det[0].xyxy[0])  # det.xyxy gives the box coordinates
#             bb_azalera = (x2 - x1) * (y2 - y1)
#             detekzio_irudia = None
#             if bb_azalera > mozketa_azalera:
#                 #bounding box bigenough. If object is car den size should be bigger than...
#                 detekzio_irudia = cv2.rectangle(predikzio_emaitza.orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(detekzio_irudia, predikzio_emaitza.names[int(det.cls)], (x1, y1 +20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#                 cv2.putText(detekzio_irudia, str(round(float(det.conf),2)), (x1, y1 +50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#                 cv2.imwrite(irteera_helbide_osoa, detekzio_irudia)


