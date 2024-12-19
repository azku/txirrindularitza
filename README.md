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

Programak, helburu bezala konfiguratu daitekeen direktorio batean irudiak iraulita kopiatzen ditu. Ondoren objektuak identifikatzeko modeloa aplikatzen zaie eta irudi bakoitzaren emaitzak ``detekzio_emaitzak.csv`` izeneko fitxategian utziko dira helburu direktorio horretan. Irudiei detekzioak marraztuko zaizkie.

## Matrikula entrenamendua

