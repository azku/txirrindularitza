
import os
import pandas as pd
from glob import glob
import shutil
from ultralytics import YOLO

DATA_PATH = "/home/ir_inf/02_13_entrenamendua_OBB"
TRAINING_PATH = "/home/ir_inf/entrenamendua_dataset_OBB"

def entrenatu():
    # Load a model
    #model = YOLO("yolov8m.pt")
    model = YOLO("yolov8n-obb.pt")

    # Train the model
    #results = model.train(data="../data/matrikulak_entrenatzeko/data.yaml",  epochs=50, task="detect", workers=1)
    #results = model.train(data="/home/ir_inf/data/matrikulak_entrenatzeko_gureak2/data.yaml",  epochs=20)
    #results = model.train(data="/home/ir_inf/data/License Plate.v8i.yolov8/data.yaml",  epochs=50)
    results = model.train(data=f"{TRAINING_PATH}/data.yaml",  epochs=200)

def label_studio_export_zatitu():

    # Load exported data
    txts = glob(DATA_PATH + '/**/*.txt')
    images = glob(DATA_PATH + '/**/*.jpg') + glob(DATA_PATH + '/**/*.png')

    # Create DataFrame
    df = pd.DataFrame({'txt': txts, 'image': images})

    # Shuffle and split data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_size = int(0.8 * len(df))
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    val_df = test_df.sample(frac=0.5)
    test_df = test_df.drop(val_df.index)

    # Create directories
    for split in ['train', 'test', 'val']:
        os.makedirs(f'{TRAINING_PATH}/{split}/images', exist_ok=True)
        os.makedirs(f'{TRAINING_PATH}/{split}/labels', exist_ok=True)

    # Copy files to respective directories
    for split, split_df in [('train', train_df), ('test', test_df), ('val', val_df)]:
        for _, row in split_df.iterrows():
            shutil.copy(row['image'], f'{TRAINING_PATH}/{split}/images')
            shutil.copy(row['txt'], f'{TRAINING_PATH}/{split}/labels')
