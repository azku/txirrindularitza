# txirrindularitza
Txirrindularitza kutxa beltzak bidalitako argazkiak aztertzeko Python bitartez.

Programaren lehenengo bertsion honek, direktorio batean dauden argazkiak aztertzen ditu eta bertan ondorengo objektuak identifikatzen saiatzen da:
    - Pertsona 0 balioa
    - Bizikleta 1 balioa
    - Kotxea 2 balioa
    - Motorra 3 balioa
    - Autobusa 5 balioa
    - Kamioia 7 balioa

Aurreko zerrendan izendatutako obketuen bat aurkitu ezkero, ``detekzio_emaitzak.csv`` izeneko fitxategi batean ateratzen dira emaitzak.
Emaitzetan, irudiaren helbidea, balio mota, konfiantza eta objectuaren detekzio mugak aterako dira.

## Aurre-ezarpenak
Argazki originalak direktorio batean egon beharko dira. Kode hau exekutatzeko, python ingurune bat izan beharko dugu  Ultralyticsen YOLO v8 eta ``imutils`` paketeak instalatuta daudelarik. 

## Prozesu orokorra

Ideia nagusia lehenengo ibilgailuak detektatzea litzateke. Detekzioak dauden irudietan matrikula detekzioa erabili genezake. Azken hau OBB oriented Bounding Box erabiliko luke eta hau ez da egokia gero matrikula irakurtzeko. Beraz, OBB emaitza horietatik abiatuz perspektiba transformazioa egin beharko dugu orientazio zuzena izan dezan koadroak. Azkenik, emaitza horretan OCR egingo genuke.
