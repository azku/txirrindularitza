import urllib.request
from ultralytics import YOLO
import cv2
import os
import pandas as pd
import imutils 
from pathlib import Path
import numpy as np
import irudi_eraldaketak

<<<<<<< HEAD
#OINARRI_DIREKTORIOA = '/home/ir_inf/data_02_07'
OINARRI_DIREKTORIOA = '/home/ir_inf/multzoa_3'
#IRTEERA_DIREKTORIO_IZENA = 'buelta_emanda'
IRTEERA_DIREKTORIO_IZENA = 'emaitzak_obb'
MODELO_IZENA = 'yolov8m.pt'
MODELO_IZENA_OBB = 'txirrindularitza_600.pt'

=======
OINARRI_DIREKTORIOA = f'{os.path.expanduser('~')}/data'
IRTEERA_DIREKTORIO_IZENA = 'emaitzak'
IRTEERA_DIREKTORIO_IZENA_OBB = 'emaitzak_obb'
MODELO_IZENA = 'yolov8m.pt'
MODELO_IZENA_OBB = 'license_plate_detector.pt'
>>>>>>> 0d9b8cae651b3624dbfce3f81042d4128286b1b1

def argazkiei_buelta_eman():
    irudi_eraldaketak.argazkiei_buelta_eman(OINARRI_DIREKTORIOA, OINARRI_DIREKTORIOA + "/buelta_emanda")

def buelta_emandako_predikzioak_margoztu(modeloa = MODELO_IZENA):
<<<<<<< HEAD
    predikzio_hiztegia = irudi_eraldaketak.predikzioak_burutu(OINARRI_DIREKTORIOA  , modeloa)
    irudi_eraldaketak.predikzioak_margoztu(predikzio_hiztegia, OINARRI_DIREKTORIOA , IRTEERA_DIREKTORIO_IZENA)
def buelta_emandako_predikzioak_margoztu_obb(modeloa = MODELO_IZENA_OBB):
    predikzio_hiztegia = irudi_eraldaketak.predikzioak_burutu_obb(OINARRI_DIREKTORIOA  , modeloa)
    irudi_eraldaketak.predikzioak_margoztu_obb(predikzio_hiztegia, OINARRI_DIREKTORIOA , IRTEERA_DIREKTORIO_IZENA)
=======
    predikzio_hiztegia = irudi_eraldaketak.predikzioak_burutu(OINARRI_DIREKTORIOA , modeloa)
    irudi_eraldaketak.predikzioak_margoztu(predikzio_hiztegia, OINARRI_DIREKTORIOA, IRTEERA_DIREKTORIO_IZENA)
>>>>>>> 0d9b8cae651b3624dbfce3f81042d4128286b1b1

def buelta_emandako_predikzioak_margoztu_obb(modeloa = MODELO_IZENA_OBB):
    predikzio_hiztegia = irudi_eraldaketak.predikzioak_burutu_obb(OINARRI_DIREKTORIOA , modeloa)
    irudi_eraldaketak.predikzioak_margoztu_obb(predikzio_hiztegia, OINARRI_DIREKTORIOA, IRTEERA_DIREKTORIO_IZENA_OBB)

def buelta_emandako_predikzioak_esportatu():
<<<<<<< HEAD
    #predikzio_hiztegia = irudi_eraldaketak.predikzioak_burutu(OINARRI_DIREKTORIOA, MODELO_IZENA)
    predikzio_hiztegia = irudi_eraldaketak.predikzioak_burutu_obb(OINARRI_DIREKTORIOA , MODELO_IZENA_OBB)
=======
    predikzio_hiztegia = irudi_eraldaketak.predikzioak_burutu(OINARRI_DIREKTORIOA, MODELO_IZENA)
>>>>>>> 0d9b8cae651b3624dbfce3f81042d4128286b1b1
    irteerako_direktorio_helbidea = OINARRI_DIREKTORIOA + "/" + IRTEERA_DIREKTORIO_IZENA 
    Path(irteerako_direktorio_helbidea).mkdir(parents=True, exist_ok=True)
    irudi_eraldaketak.predikzioak_dataframe_bihurtu(predikzio_hiztegia).to_csv(irteerako_direktorio_helbidea + "/detekzio_emaitza.csv")

def buelta_emandako_predikzioak_esportatu_obb():
    predikzio_hiztegia = irudi_eraldaketak.predikzioak_burutu_obb(OINARRI_DIREKTORIOA, MODELO_IZENA_OBB)
    irteerako_direktorio_helbidea = OINARRI_DIREKTORIOA + "/" + IRTEERA_DIREKTORIO_IZENA_OBB 
    Path(irteerako_direktorio_helbidea).mkdir(parents=True, exist_ok=True)
    irudi_eraldaketak.predikzioak_dataframe_bihurtu(predikzio_hiztegia).to_csv(irteerako_direktorio_helbidea + "/detekzio_emaitza.csv")
