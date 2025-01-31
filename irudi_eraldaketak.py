import os
import imutils 
import cv2
from pathlib import Path
from itertools import chain
from ultralytics import YOLO


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

def argazkiei_buelta_eman(sarrera_direktorioa, irteera_izena):
    for fitxategi_helbide_osoa in get_direktorioko_irudiak(sarrera_direktorioa):
        argazkiari_buelta_eman(fitxategi_helbide_osoa,  sarrera_direktorioa + "/" + irteera_izena)

def argazkiari_buelta_eman(fitxategi_helbide_osoa, irteera_direktorioa):
    burua_fitxategia = os.path.split(fitxategi_helbide_osoa)
    helburu_bide_izena = irteera_direktorioa + "/" + burua_fitxategia[1]
    irudia = cv2.imread(fitxategi_helbide_osoa, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
    iraulitako_irudia =    imutils.rotate(irudia, angle=180)
    cv2.imwrite(helburu_bide_izena, iraulitako_irudia)

def predikzioak_burutu(sarrera_direktorioa, modeloaren_izena, mozketa_azalera=200000):
    model = YOLO(modeloaren_izena)
    def predikzioa_burutu(fitxategi_helbide_osoa):
        predikzio_emaitza = model.predict(fitxategi_helbide_osoa, verbose=False)[0]
        for det in  predikzio_emaitza.boxes:
            if int(det.cls)  in [0, 1, 2, 3, 5, 7 ]:
                x1, y1, x2, y2 = map(int, det[0].xyxy[0])  # det.xyxy gives the box coordinates
                bb_azalera = (x2 - x1) * (y2 - y1)
                yield {"fitxategi_helbide_osoa":fitxategi_helbide_osoa,
                       "irudi_originala": predikzio_emaitza.orig_img,
                       "konfiantza": float(det.conf),
                       "etiketa":  predikzio_emaitza.names[int(det.cls)],
                       "x1": x1, "y1":y1, "x2": x2, "y2":y2,
                       "bb_azalera": bb_azalera}
    return chain.from_iterable(map(predikzioa_burutu, get_direktorioko_irudiak(sarrera_direktorioa)))

def predikzioa_margoztu(irudi_originala, x1, y1, x2, y2, etiketa, konfiantza, irteera_helbidea):
    detekzio_irudia = cv2.rectangle(irudi_originala, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(detekzio_irudia, etiketa, (x1, y1 +20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(detekzio_irudia, str(round(konfiantza,2)), (x1, y1 +50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imwrite(irteera_helbidea, detekzio_irudia)

def predikzioak_margoztu(predikzio_hiztegia, sarrera_direktorio_helbidea, irteera_direktorio_izena, konfiantza_minimoa=0.4):
    for p in predikzio_hiztegia:
        if p["konfiantza"]>0.4:
            helbide_zatitua = os.path.split(p["fitxategi_helbide_osoa"])
            helburu_bide_izena = sarrera_direktorio_helbidea + "/" + irteera_direktorio_izena + "/" + helbide_zatitua[1]
            predikzioa_margoztu(p['irudi_originala'], p["x1"], p["y1"], p["x2"], p["y2"], p["etiketa"], p["konfiantza"], helburu_bide_izena)

def predikzioak_dataframe_bihurtu(predikzio_hiztegia):
    metadatu_hiztegia = arg_meta.get_picture_metadata()
    df_lista = []
    for p in predikzio_hiztegia:
        helbide_zatitua = os.path.split(p["fitxategi_helbide_osoa"])
        ml, rd, dist = identifikatu_izeneko_datuak(helbide_zatitua[1])
        metadatuak = metadatu_hiztegia[helbide_zatitua[1]]
        df_lista.append({"id": metadatuak["id"], "date": metadatuak["date"], "time": metadatuak["time"], "route": metadatuak["route"],
                         'miliseconds': ml, 'distance': dist, 'image_path': p["fitxategi_helbide_osoa"],
                         'confidence':str(p["konfiantza"]) , 'type': p["etiketa"], 'bb_x1': p["x1"], 'bb_y1': p["x2"],
                         'bb_x2': p["x2"], 'bb_y2': p["y2"], 'area': (p["x2"] - p["x1"]) * (p["y2"] - p["y1"])})
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


