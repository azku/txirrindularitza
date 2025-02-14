import urllib.request
from ultralytics import YOLO
import cv2
import os
import pandas as pd
import imutils 
from pathlib import Path
import numpy as np
import argazkien_metadatoak_ekarri as arg_meta
import irudi_eraldaketak

#OINARRI_DIREKTORIOA = '/home/ir_inf/data_02_07'
OINARRI_DIREKTORIOA = '/home/ir_inf/02_07_entrenamendua/images'
#IRTEERA_DIREKTORIO_IZENA = 'buelta_emanda'
IRTEERA_DIREKTORIO_IZENA = 'emaitzak_obb'
MODELO_IZENA = 'license_plate_detector.pt'
#MODELO_IZENA = 'yolov8m.pt'

def identifikatu_izeneko_datuak(filename):
    s = filename.split('_')
    return (s[1],s[2],s[3]) #miliseconds, random, distance

def argazkiei_buelta_eman():
    irudi_eraldaketak.argazkiei_buelta_eman(OINARRI_DIREKTORIOA, IRTEERA_DIREKTORIO_IZENA)


def buelta_emandako_predikzioak_margoztu(modeloa = MODELO_IZENA):
    predikzio_hiztegia = irudi_eraldaketak.predikzioak_burutu(OINARRI_DIREKTORIOA , modeloa)
    irudi_eraldaketak.predikzioak_margoztu(predikzio_hiztegia, OINARRI_DIREKTORIOA, IRTEERA_DIREKTORIO_IZENA)
def buelta_emandako_predikzioak_margoztu_obb(modeloa = MODELO_IZENA):
    predikzio_hiztegia = irudi_eraldaketak.predikzioak_burutu_obb(OINARRI_DIREKTORIOA , modeloa)
    irudi_eraldaketak.predikzioak_margoztu_obb(predikzio_hiztegia, OINARRI_DIREKTORIOA, IRTEERA_DIREKTORIO_IZENA)


def buelta_emandako_predikzioak_esportatu():
    predikzio_hiztegia = irudi_eraldaketak.predikzioak_burutu(OINARRI_DIREKTORIOA, MODELO_IZENA)
    irudi_eraldaketak.predikzioak_dataframe_bihurtu(predikzio_hiztegia).to_csv(OINARRI_DIREKTORIOA + "/" + IRTEERA_DIREKTORIO_IZENA + "/detekzio_emaitza.csv")
