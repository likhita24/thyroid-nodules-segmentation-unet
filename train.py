import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from unet_model import UNET
import matplotlib.pyplot as plt
import cv2
from PIL import Image

from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

import os

# Hyperparameters etc.
LEARNING_RATE = 1e-4
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
device = "4" # Put the GPU ID here 

# .to(device)

# .to("cuda:6")
# .cuda()

BATCH_SIZE = 1
NUM_EPOCHS = 5
NUM_WORKERS = 2
IMAGE_HEIGHT = 160 
IMAGE_WIDTH = 240
PIN_MEMORY = True
LOAD_MODEL = False

TRAIN_IMG_DIR = "/DATA/bitra1/data_divs/train/data/"
TRAIN_MASK_DIR = "/DATA/bitra1/mask_divs/train/"
VAL_IMG_DIR = "/DATA/bitra1/data_divs/val/data/"
VAL_MASK_DIR = "/DATA/bitra1/mask_divs/val/"
TEST_IMG_DIR = "/DATA/bitra1/data_divs/test/data/"
TEST_MASK_DIR = "/DATA/bitra1/mask_divs/test/"


os.environ['CUDA_VISIBLE_DEVICES'] = device

# def display_image_grid(images_filenames, images_directory, masks_directory, predicted_masks=None):
#     cols = 3 if predicted_masks else 2
#     rows = len(images_filenames)
#     figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(10, 24))
#     for i, image_filename in enumerate(images_filenames):
#         print("image filename : ", image_filename)
#         print(images_directory + image_filename)

#         image = cv2.imread((images_directory + image_filename))
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         print(type(image))
#         mask = cv2.imread(os.path.join(masks_directory, image_filename.replace(".jpg", ".png")), cv2.IMREAD_UNCHANGED,)
#         #image = open_image(images_directory + image_filename)
#         ax[i, 0].imshow(image)
#         ax[i, 1].imshow(mask, interpolation="nearest")

#         ax[i, 0].set_title("Image")
#         ax[i, 1].set_title("Ground truth mask")

#         ax[i, 0].set_axis_off()
#         ax[i, 1].set_axis_off()

#         if predicted_masks:
#             predicted_mask = predicted_masks[i]
#             ax[i, 2].imshow(int(predicted_mask), interpolation="nearest")
#             ax[i, 2].set_title("Predicted mask")
#             ax[i, 2].set_axis_off()
#     plt.tight_layout()
#     plt.show()

def train_fn(loader, model, optimizer, loss_fn, scaler):
    step = 0
    for epoch in range(NUM_EPOCHS):
        loop = tqdm(loader)
        
        for batch_idx, (data, targets, s) in enumerate(loop):
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

# def plot_image(preds):
#     figure, ax = plt.subplots(nrows=len(preds), ncols=2, figsize=(10, 24))
#     for i in range(len(preds)):
#         mask = preds[i]
#         ax[i, 0].imshow(mask.cpu().squeeze().numpy())
#         #ax[i, 1].imshow(mask, interpolation="nearest")
#         ax[i, 0].set_title("Augmented image")
#         #ax[i, 1].set_title("Augmented mask")
#         ax[i, 0].set_axis_off()
#         #ax[i, 1].set_axis_off()
#     plt.tight_layout()
#     plt.show()

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

    test_transforms = A.Compose(
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

    train_loader, val_loader, test_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        TEST_IMG_DIR,
        TEST_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        test_transforms,
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
        save_predictions_as_imgs("/DATA/bitra1/mask_images/val/", val_loader, model, folder="/DATA/bitra1/val_saved_images")
    

    print("\n---------------------------Test accuracy---------------------------------\n")

    check_accuracy(test_loader, model)
    save_predictions_as_imgs("/DATA/bitra1/mask_images/test/", test_loader, model, folder="/DATA/bitra1/test_saved_images")
        


if __name__ == "__main__":
    main()