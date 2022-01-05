#import splitfolders 
import os

#splitfolders.ratio('/DATA/bitra1/main_dataset', output='/DATA/bitra1/data_divs/', seed=1337, ratio=(0.7, 0.2, 0.1), group_prefix = None)

path, dirs, files = next(os.walk('/DATA/bitra1/mask_divs/val'))
file_count = len(files)
print(file_count)