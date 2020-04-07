import glob
import os

import cv2
from tqdm import tqdm


model = 'SALICON'

# Path to SALICON raw data
DIR_raw_img_train = '../data/SALICON/images/train'
DIR_raw_map_train = '../data/SALICON/maps/train'
DIR_raw_img_val = '../data/SALICON/images/val'
DIR_raw_map_val = '../data/SALICON/maps/val'

# Path to processed data
DIR_resized_img_train = ''.join(['../data/', model, '/imgs_train'])
DIR_resized_map_train = ''.join(['../data/', model, '/maps_train'])
DIR_resized_img_train = ''.join(['../data/', model, '/imgs_val'])
DIR_resized_map_val = ''.join(['../data/', model, '/maps_val'])

INPUT_SIZE = (256, 192)
error_img_list = []


if not os.path.exists(DIR_resized_img_train):
    os.makedirs(DIR_resized_img_train)
if not os.path.exists(DIR_resized_map_train):
    os.makedirs(DIR_resized_map_train)
if not os.path.exists(DIR_resized_img_train):
    os.makedirs(DIR_resized_img_train)
if not os.path.exists(DIR_resized_map_val):
    os.makedirs(DIR_resized_map_val)


list_img_files = [k.split(os.sep)[-1].split('.')[0] for k in glob.glob(os.path.join(DIR_raw_img_train, '*train*'))]
print(len(list_img_files))

for curr_file in tqdm(list_img_files):
    pre_img_name = ''.join([curr_file, '.jpg'])
    post_img_name = ''.join([curr_file, '.png'])
    
    full_img_path = os.path.join(DIR_raw_img_train, pre_img_name)
    try:
        imageResized = cv2.resize(cv2.imread(full_img_path), INPUT_SIZE, interpolation=cv2.INTER_AREA)
            
        full_map_path = os.path.join(DIR_raw_map_train, post_img_name)
        mapResized = cv2.resize(cv2.imread(full_map_path), INPUT_SIZE, interpolation=cv2.INTER_AREA)
        
        cv2.imwrite(os.path.join(DIR_resized_img_train, post_img_name), imageResized)
        cv2.imwrite(os.path.join(DIR_resized_map_train, post_img_name), mapResized)
    except:
        print('Error')
        error_img_list.append(curr_file)
print(error_img_list)


list_img_files = [k.split(os.sep)[-1].split('.')[0] for k in glob.glob(os.path.join(DIR_raw_img_val, '*val*'))]
print(len(list_img_files))

for curr_file in tqdm(list_img_files):
    pre_img_name = ''.join([curr_file, '.jpg'])
    post_img_name = ''.join([curr_file, '.png'])

    full_img_path = os.path.join(DIR_raw_img_val, pre_img_name)
    imageResized = cv2.resize(cv2.imread(full_img_path), INPUT_SIZE, interpolation=cv2.INTER_AREA)
        
    full_map_path = os.path.join(DIR_raw_map_val, post_img_name)
    mapResized = cv2.resize(cv2.imread(full_map_path), INPUT_SIZE, interpolation=cv2.INTER_AREA)
    
    cv2.imwrite(os.path.join(DIR_resized_img_train, post_img_name), imageResized)
    cv2.imwrite(os.path.join(DIR_resized_map_val, post_img_name), mapResized)
    

print('Done resizing images.')