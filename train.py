import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from unet_model import UNET

from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

import os

# Hyperparameters etc.
LEARNING_RATE = 1e-0
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
device = "3" # Put the GPU ID here 

# .to(device)

# .to("cuda:6")
# .cuda()

BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 288 
IMAGE_WIDTH = 432
PIN_MEMORY = True
LOAD_MODEL = False

TRAIN_IMG_DIR = "/DATA/bitra1/data_divs/train/data"
TRAIN_MASK_DIR = "/DATA/bitra1/mask_divs/train"
VAL_IMG_DIR = "/DATA/bitra1/data_divs/val/data"
VAL_MASK_DIR = "/DATA/bitra1/mask_divs/val"
TEST_IMG_DIR = "/DATA/bitra1/data_divs/test"
TEST_VAL_DIR = "/DATA/bitra1/mask_divs/test"


os.environ['CUDA_VISIBLE_DEVICES'] = device

def train_fn(loader, model, optimizer, loss_fn, scaler):
    step = 0
    for epoch in range(NUM_EPOCHS):
        loop = tqdm(loader)
        print(len(loop))
        for batch_idx, (data, targets) in enumerate(loop):
            data = data.cuda()
            targets = targets.float().unsqueeze(1).cuda()
            # forward
            with torch.cuda.amp.autocast():
                # if(c == 0):
                #     print(data, ", length = ", len(data))
                predictions = model(data)
                # if(c == 0):
                #     print(predictions, ", predictions = ", len(predictions))
                loss = loss_fn(predictions, targets)

            # backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # update tqdm loop
            loop.set_postfix(loss=loss.item())
            step += 1
        check_accuracy(loader, model)


def main():
    train_transform = A.Compose(
        
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNET(in_channels=3, out_channels=1).cuda()
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)


    check_accuracy(val_loader, model, device=device)
    
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }

        save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy(val_loader, model)

        # print some examples to a folder
        save_predictions_as_imgs(val_loader, model, folder="/DATA/bitra1/saved_images")

if __name__ == "__main__":
    main()