import pandas as pd
import numpy as np
import torch
import cv2
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

BASE_PATH = '../input/uw-madison-gi-tract-image-segmentation'
MASK_PATH = '../input/uwmgipreprocessed'

df = pd.read_csv(f'{MASK_PATH}/preprocessed_train.csv')
df[df['empty'] == False]

def rle2mask(mask_rle: str, shape: np.ndarray, label=1):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = label
    return img.reshape(shape)
    
class Dataset(Dataset):
    def __init__(self, images, masks, device):
        self.images = images
        self.masks = masks
        self.device = device

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        mask_path = self.masks[index]
        
        # loading images
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        img = np.tile(img[...,None], [1, 1, 3])
        img = img.astype('float32')
        mx = np.max(img)
        if mx:
            img/=mx
        
        # loading masks
        mask = np.load(mask_path)
        
        
        return img, mask
        
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset = Dataset(df['image_path'], df['mask_path'], device)
im = dataset.__getitem__(83)

#plt.imshow(rle2mask(im[1][2], im[0].shape))
print(im[0].shape)
plt.imshow(im[0])
