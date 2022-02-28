"""
作用:把train中的数据随机分一部分到test中(包括label跟image)
日期:20201118
"""
import os
import sys
from random import shuffle
import shutil
from tqdm import tqdm

TRAIN_IMG_PATH = '/emwuser/znr/data/GF3_single_dataset/img_val'
TRAIN_LAB_PATH = '/emwuser/znr/data/GF3_single_dataset/lab_val'
TEST_IMG_PATH = '/emwuser/znr/code/deeplabv3p_pytorch_znr/data/img_val'
TEST_LAB_PATH = '/emwuser/znr/code/deeplabv3p_pytorch_znr/data/lab_val'
REMOVE_RATE = 0.1

def main():
    img_list = os.listdir(TRAIN_IMG_PATH)
    shuffle(img_list)
    img_num = len(img_list)
    remove_num = int(REMOVE_RATE * img_num)
    remove_img = img_list[:remove_num]
    (_, ex_before) = os.path.splitext(remove_img[0])

    lab_list = os.listdir(TRAIN_LAB_PATH)
    remove_lab = []
    print('preparing train images and labels to remove to test......')
    for ii in tqdm(lab_list,total=len(lab_list)):
        (name, _) = os.path.splitext(ii)
        if (name + ex_before) in remove_img:
            remove_lab.append(ii)
    
    print('removing images......')
    for img in tqdm(remove_img,total=len(remove_img)):
        # shutil.move(os.path.join(TRAIN_IMG_PATH, img),os.path.join(TEST_IMG_PATH, img))
        shutil.copy(os.path.join(TRAIN_IMG_PATH, img),os.path.join(TEST_IMG_PATH, img))
    
    print('removing labels......')
    for lab in tqdm(remove_lab,total=len(remove_lab)):
        # shutil.move(os.path.join(TRAIN_LAB_PATH, lab),os.path.join(TEST_LAB_PATH, lab))
        shutil.copy(os.path.join(TRAIN_LAB_PATH, lab),os.path.join(TEST_LAB_PATH, lab))

if __name__ == "__main__":
    main()
