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
from matplotlib import pyplot as plt
from torch.utils.data.sampler import RandomSampler
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

BASE_PATH = '../input/uw-madison-gi-tract-image-segmentation'
MASK_PATH = '../input/uwmgipreprocessed'

class CFG:
    EPOCHS = 5
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    NB_CLASSES = 4
    NUM_EPOCHS = 1
    VALIDATION_SPLIT = 0.2
    BATCH_SIZE = 32
    NUM_WORKERS = 0
    SHUFFLE_DATASET = True

df = pd.read_csv(f'{MASK_PATH}/preprocessed_train.csv')
df[df['empty'] == False]

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
        #print(f'image_path: {image_path}')
        #print(f'img shape: {img.shape}')
        img = cv2.resize(img, (256, 256), 0, 0, cv2.INTER_NEAREST)
        img = np.tile(img[...,None], [1, 1, 3])
        img = img.astype('float32')
        mx = np.max(img)
        if mx:
            img/=mx
        
        # loading masks
        mask = np.load(mask_path)
        #print(f'mask_path: {mask_path}')
        #print(f'mask shape: {mask.shape}')
        mask = cv2.resize(mask, (256, 256), 0, 0, cv2.INTER_NEAREST)
        mask = mask.astype(np.float32)
        
        img = img.transpose(2, 0, 1)
        mask = mask.transpose(2, 0, 1)
        
        img = torch.tensor(img).to(self.device)
        mask = torch.tensor(mask).to(self.device)
        return img, mask
        
check_dataset = Dataset(df['image_path'], df['mask_path'], CFG.DEVICE)
im = check_dataset.__getitem__(7500)#83)
print(im[0].shape)
print(im[1].shape)
print(np.unique(im[0].cpu().detach().numpy()))
print(np.unique(im[1].cpu().detach().numpy()))
plt.imshow(im[0].cpu().detach().numpy().transpose(1, 2, 0))
plt.imshow(im[1].cpu().detach().numpy().transpose(1, 2, 0)*255, alpha=0.5)
        
def double_conv(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )
    return conv


def crop_img(tensor, target_tensor):
    # square images
    target_size = target_tensor.size()[2]
    if tensor.size()[2] % 2 == 1:
        tensor_size = tensor.size()[2] - 1
    else:
        tensor_size = tensor.size()[2]
    delta = tensor_size - target_size
    delta = delta // 2
    return tensor[:, :, delta:tensor_size - delta, delta:tensor_size - delta]

class UNet(nn.Module):
    def __init__(self, nb_classes):
        super(UNet, self).__init__()
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_1 = double_conv(3, 64)
        self.down_conv_2 = double_conv(64, 128)
        self.down_conv_3 = double_conv(128, 256)
        self.down_conv_4 = double_conv(256, 512)
        self.down_conv_5 = double_conv(512, 1024)

        ## transposed convolutions
        self.up_trans_1 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.up_conv_1 = double_conv(1024, 512)

        self.up_trans_2 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.up_conv_2 = double_conv(512, 256)

        self.up_trans_3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.up_conv_3 = double_conv(256, 128)

        self.up_trans_4 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.up_conv_4 = double_conv(128, 64)

        self.out = nn.Conv2d(64, nb_classes, 1)


    def forward(self, image):
        # encoder part
        # input image
        x1 = self.down_conv_1(image) # this is passed to decoder
        # max pooling
        x2 = self.max_pool_2x2(x1)

        x3 = self.down_conv_2(x2) # this is passed to decoder
        x4 = self.max_pool_2x2(x3)

        x5 = self.down_conv_3(x4) # this is passed to decoder
        x6 = self.max_pool_2x2(x5)

        x7 = self.down_conv_4(x6) # this is passed to decoder
        x8 = self.max_pool_2x2(x7)

        x9 = self.down_conv_5(x8)

        # decoder part
        x = self.up_trans_1(x9)
        y = crop_img(x7, x)
        x = self.up_conv_1(torch.cat([x, y], 1))

        x = self.up_trans_2(x)
        y = crop_img(x5, x)
        x = self.up_conv_2(torch.cat([x, y], 1))

        x = self.up_trans_3(x)
        y = crop_img(x3, x)
        x = self.up_conv_3(torch.cat([x, y], 1))

        x = self.up_trans_4(x)
        y = crop_img(x1, x)
        x = self.up_conv_4(torch.cat([x, y], 1))

        x = self.out(x)
        return x

df = df[:5000]
dataset = Dataset(df['image_path'], df['mask_path'], CFG.DEVICE)
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(CFG.VALIDATION_SPLIT * dataset_size))
if CFG.SHUFFLE_DATASET:
    np.random.seed(2022)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]
print(len(train_indices), len(val_indices))
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=CFG.BATCH_SIZE, sampler=train_sampler, num_workers=CFG.NUM_WORKERS)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=CFG.BATCH_SIZE,sampler=valid_sampler, num_workers=CFG.NUM_WORKERS)


model = UNet(3)
model = model.to(CFG.DEVICE)
criterion = torch.nn.modules.loss.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


def train_epoch():
    train_bar = tqdm(train_loader)
    train_loss = []
    correct = 0
    for (data, target) in train_bar:
        print(data.shape, target.shape)
        model.train()
        data, target = data.to(CFG.DEVICE), target.to(CFG.DEVICE)
        optimizer.zero_grad()
        output = model(data)
        print(output.shape)
        time.sleep(60)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()
        # saving loss
        loss = loss.item()
        train_loss.append(loss)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        train_bar.set_description('loss: %.5f, smth: %.5f' % (loss, smooth_loss))
        
def val_epoch():
    model.eval()
    val_loss = []
    correct = 0
    with torch.no_grad():
        for (data, target) in tqdm(validation_loader):
            data, target = data.to(CFG.DEVICE), target.to(CFG.DEVICE)
            logits = model(data)
            loss = criterion(logits, target)
            # saving loss
            loss = loss.item()
            val_loss.append(loss)
        val_loss = np.array(val_loss).sum() / len(validation_loader.sampler)
        
        
for epoch in range(1,3):
    print(time.ctime(), 'Epoch:', epoch)
    train_epoch()
    val_loss, dice = val_epoch()
    print("Val_Dice = {} vall_loss = {}".format(dice, val_loss))
    return val_loss, val_dice
    

torch.save(model.state_dict(), './model.pt')

model_eval = UNet(3)
model_eval.load_state_dict(torch.load('./model'))
model_eval.to(CFG.DEVICE)
model_eval.eval()

eval_check = Dataset(df['image_path'], df['mask_path'], CFG.DEVICE)
single_data, single_target = eval_check.__getitem__(100)
with torch.no_grad():
    logits = model_eval(single_data.unsqueeze(dim=0).to(CFG.DEVICE))
    y_pred = nn.Softmax()(logits)
    plt.imshow(single_data.cpu().detach().numpy().transpose(1, 2, 0))
    plt.imshow(y_pred.squeeze().cpu().detach().numpy().transpose(1, 2, 0)*255, alpha=0.5)
    print(np.unique(y_pred.squeeze().cpu().detach().numpy()))
    

eval_check = Dataset(df['image_path'], df['mask_path'], CFG.DEVICE)
single_data, single_target = eval_check.__getitem__(100)
with torch.no_grad():
    logits = model_eval(single_data.unsqueeze(dim=0).to(CFG.DEVICE))
    y_pred = nn.Softmax()(logits)
    plt.imshow(single_data.cpu().detach().numpy().transpose(1, 2, 0))
    plt.imshow(y_pred.squeeze().cpu().detach().numpy().transpose(1, 2, 0)*255, alpha=0.5)
    print(np.unique(y_pred.squeeze().cpu().detach().numpy()))
