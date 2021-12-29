import os
import shutil
        
data_train = '/DATA/bitra1/data_divs/train/data'
data_test = '/DATA/bitra1/data_divs/test/data'
data_val = '/DATA/bitra1/data_divs/val/data'

mask_dataset = '/DATA/bitra1/masks_binary/'

mask_train = '/DATA/bitra1/mask_divs/train/'
mask_test = '/DATA/bitra1/mask_divs/test/'
mask_val = '/DATA/bitra1/mask_divs/val/'

for filename in os.listdir(data_train):
    j = 0
    print("filename : ", filename)
    for f in os.listdir(mask_dataset):
        print("f : ", os.path.splitext(f)[0])
        if(os.path.splitext(f)[0] == os.path.splitext(filename)[0]):
            j = 1
            shutil.move(mask_dataset+f, mask_train, copy_function = shutil.copy2)
            break
    if(j == 0):
        print(os.path.splitext(filename)[0])

for filename in os.listdir(data_test):
    j = 0
    print("filename : ", filename)
    for f in os.listdir(mask_dataset):
        print("f : ", os.path.splitext(f)[0])
        if(os.path.splitext(f)[0] == os.path.splitext(filename)[0]):
            j = 1
            shutil.move(mask_dataset+f, mask_test, copy_function = shutil.copy2)
            break
    if(j == 0):
        print(os.path.splitext(filename)[0])

for filename in os.listdir(data_val):
    j = 0
    print("filename : ", filename)
    for f in os.listdir(mask_dataset):
        print("f : ", os.path.splitext(f)[0])
        if(os.path.splitext(f)[0] == os.path.splitext(filename)[0]):
            j = 1
            shutil.move(mask_dataset+f, mask_val, copy_function = shutil.copy2)
            break
    if(j == 0):
        print(os.path.splitext(filename)[0])


