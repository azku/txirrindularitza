import urllib.request
from ultralytics import YOLO
import cv2
import os
import pandas as pd
import imutils 
from pathlib import Path
import numpy as np
import irudi_eraldaketak

#OINARRI_DIREKTORIOA = '/home/ir_inf/data_02_07'
OINARRI_DIREKTORIOA = '/home/ir_inf/data/otsaileko_398'
#IRTEERA_DIREKTORIO_IZENA = 'buelta_emanda'
IRTEERA_DIREKTORIO_IZENA = 'emaitzak_obb'
#MODELO_IZENA = 'license_plate_detector.pt'
MODELO_IZENA = 'yolo11m-obb.pt'


def argazkiei_buelta_eman():
    irudi_eraldaketak.argazkiei_buelta_eman(OINARRI_DIREKTORIOA, IRTEERA_DIREKTORIO_IZENA)


def buelta_emandako_predikzioak_margoztu(modeloa = MODELO_IZENA):
    predikzio_hiztegia = irudi_eraldaketak.predikzioak_burutu(OINARRI_DIREKTORIOA , modeloa)
    irudi_eraldaketak.predikzioak_margoztu(predikzio_hiztegia, OINARRI_DIREKTORIOA, IRTEERA_DIREKTORIO_IZENA)
def buelta_emandako_predikzioak_margoztu_obb(modeloa = MODELO_IZENA):
    predikzio_hiztegia = irudi_eraldaketak.predikzioak_burutu_obb(OINARRI_DIREKTORIOA , modeloa)
    irudi_eraldaketak.predikzioak_margoztu_obb(predikzio_hiztegia, OINARRI_DIREKTORIOA, IRTEERA_DIREKTORIO_IZENA)


def buelta_emandako_predikzioak_esportatu():
    #predikzio_hiztegia = irudi_eraldaketak.predikzioak_burutu(OINARRI_DIREKTORIOA, MODELO_IZENA)
    predikzio_hiztegia = irudi_eraldaketak.predikzioak_burutu_obb(OINARRI_DIREKTORIOA, MODELO_IZENA)
    irteerako_direktorio_helbidea = OINARRI_DIREKTORIOA + "/" + IRTEERA_DIREKTORIO_IZENA 
    Path(irteerako_direktorio_helbidea).mkdir(parents=True, exist_ok=True)
    irudi_eraldaketak.predikzioak_dataframe_bihurtu(predikzio_hiztegia).to_csv(irteerako_direktorio_helbidea + "/detekzio_emaitza.csv")
