import numpy as np
import pandas as pd
import os
from glob import glob
import cv2

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image

DF_CONSOLE_WIDTH = 500
pd.set_option('display.width', DF_CONSOLE_WIDTH)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
DATA_PATH = 'data/train'
IMAGE_SIZE = 256


class Dataset(Dataset):
    def __init__(self, images, masks, device):
        self.images = images
        self.masks = masks
        self.device = device

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        mask = self.masks[index]

        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, (256,256), interpolation=cv2.INTER_AREA)
        img = np.tile(img[..., None], [1,1,3])
        img = img.astype('float32')

        mask = np.asarray(mask, dtype="float32")
        mask = cv2.resize(mask, (256,256), interpolation=cv2.INTER_AREA)

        img /= 255
        img = img.transpose(2, 0, 1)

        mask /= 255

        img = torch.tensor(img).to(self.device)
        mask = torch.tensor(mask).to(self.device)
        return img, mask

def rle2mask(mask_rle: str, shape: np.ndarray, label=1):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = label
    return img.reshape(shape)


def create_path_to_slice(root_dir, case_id):
    case_folder = case_id.split("_")[0]
    day_folder = "_".join(case_id.split("_")[:2])
    file_starter = "_".join(case_id.split("_")[2:])
    # fetching folder paths
    folder = os.path.join(root_dir, case_folder, day_folder, "scans")
    # fetching filenames with similar pattern
    file = glob(f"{folder}/{file_starter}*")
    # returning the first file, though it will always hold one file.
    return file[0]


def preprocess():
    df = pd.read_csv('data/train.csv')
    df['segmentation'] = df['segmentation'].astype('str')
    df['case_id'] = df['id'].apply(lambda x: x.split("_")[0][4:])
    df['day_id'] = df['id'].apply(lambda x: x.split("_")[1][3:])
    df['slice_id'] = df['id'].apply(lambda x: x.split("_")[-1])
    df['path'] = df['id'].apply(lambda x: create_path_to_slice(DATA_PATH, x))
    df['height'] = df['path'].apply(lambda x: os.path.split(x)[-1].split("_")[2]).astype("int")
    df['width'] = df['path'].apply(lambda x: os.path.split(x)[-1].split("_")[3]).astype("int")
    class_names = df['class'].unique()
    for index, label in enumerate(class_names):
        df['class'].replace(label, index, inplace=True)
    df.to_csv('data/preprocessed_train.csv')
