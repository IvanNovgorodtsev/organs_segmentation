import numpy as np
import pandas as pd
import os
import shutil
from glob import glob
import cv2
import torch.nn as nn
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import time
from numpy import asarray
from numpy import save

BASE_PATH  = '../input/uw-madison-gi-tract-image-segmentation'
MASK_PATH  = '../input/uwmgipreprocessed'

def ignore_files(dir, files):
    return [f for f in files if os.path.isfile(os.path.join(dir, f))]
 
shutil.copytree(f'{BASE_PATH}/train', './mask', ignore=ignore_files)

df = pd.read_csv(f'{BASE_PATH}/train.csv')
df['case'] = df['id'].str.split('_', 0, expand=True)[0]
df['day'] = df['id'].str.split('_', 0, expand=True)[1]
df['slice'] = df['id'].str.split('_', 0, expand=True)[3]
df['segmentation'] = df['segmentation'].fillna('')
df['rle_len'] = df['segmentation'].map(len)
df_temp = df.groupby(['id'])['segmentation'].agg(list).to_frame().reset_index()
df_temp = df_temp.merge(df.groupby(['id'])['rle_len'].agg(sum).to_frame().reset_index())
df = df.drop(columns=['segmentation', 'class', 'rle_len'])
df = df.groupby(['id']).head(1).reset_index(drop=True)
df = df.merge(df_temp, on=['id'])
# empty masks
df['empty'] = (df['rle_len']==0)
# image paths
df['image_path'] = df.apply(lambda row: f'{BASE_PATH}/train/{row.case}/{row.case}_{row.day}/scans/slice_{row.slice}_266_266_1.50_1.50.png', axis=1)
# saving numpy arrays and their paths
df['mask_path'] = df.apply(lambda row: f'{MASK_PATH}/masks/{row.case}/{row.case}_{row.day}/scans/slice_{row.slice}_266_266_1.50_1.50.npy', axis=1)
df.apply(lambda row: save(f'./mask/{row.case}/{row.case}_{row.day}/scans/slice_{row.slice}_266_266_1.50_1.50.npy', asarray(row['segmentation'])), axis=1)
df.to_csv('preprocessed_train.csv')
