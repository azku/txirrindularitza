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
#OINARRI_DIREKTORIOA = '/home/asier/Documents/txirrindularitza/Irudiak 25_02_2026'
OINARRI_DIREKTORIOA = "/home/asier/Documents/txirrindularitza/Irudiak 09_03_2026"
#IRTEERA_DIREKTORIO_IZENA = 'buelta_emanda'
IRTEERA_DIREKTORIO_IZENA = 'emaitzak_ocr'
MODELO_IZENA = 'license_plate_detector.pt'





predikzio_hiztegia = irudi_eraldaketak.predikzioak_burutu_obb(OINARRI_DIREKTORIOA, MODELO_IZENA)
irteerako_direktorio_helbidea = OINARRI_DIREKTORIOA + "/" + IRTEERA_DIREKTORIO_IZENA 
Path(irteerako_direktorio_helbidea).mkdir(parents=True, exist_ok=True)
irudi_eraldaketak.predikzioak_margoztu_obb(predikzio_hiztegia, OINARRI_DIREKTORIOA , IRTEERA_DIREKTORIO_IZENA)



