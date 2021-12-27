"""
    author:nrzheng
    function:cutting pic
    edition:final
    date:2021.12.15
"""

import os
import argparse
import glob
from PIL import Image
import numpy as np
import math
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None

def parse_args():
    """
        参数设置
    """
    parser = argparse.ArgumentParser(description='cutting pic')
    parser.add_argument('--path', help='the path of all wave band file', default=r'D:\项目\data\dataset\datasets')
    parser.add_argument('--wave_band', help='the cutting band', default='S')
    parser.add_argument('--size', help='the cutting size', default=512)
    parser.add_argument('--strides', help='the moving strides', default=256)

    args = parser.parse_args()
    return args

def pic_cutting(args, images, save_path):
    """
        切图代码
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for image in images:
        image_names = os.path.split(image)[1]
        image_name, ext = os.path.splitext(image_names)
        img = Image.open(image)
        img = np.asarray(img)
        h, w, c = img.shape
        stride = args.strides
        height = args.size
        width = args.size

        # 计算切图的行数和列数
        row = math.floor((h - height) / stride)
        column = math.floor((w - width) / stride)

        id_hang = 1
        for i in tqdm(range(row), total=row):
            row_start = i * stride
            row_end = i * stride + height
            id_lie = 1
            for j in range(column):
                col_start = j * stride
                col_end = j * stride + width
                small_pic = img[row_start:row_end, col_start:col_end, :]
                small_name = image_name + '_' + str(id_hang).rjust(3, '0') + 'row_' + str(id_lie).rjust(3, '0') + 'col' + ext
                small_pic = Image.fromarray(np.uint8(small_pic))
                small_pic.save(os.path.join(save_path, small_name))
                id_lie += 1
            id_hang += 1



def main():
    """
        主函数
    """
    args = parse_args()
    wave_band = args.wave_band
    args.train_path = os.path.join(args.path, wave_band, 'train')
    img_save_path = os.path.join(args.path, wave_band, 'img_train_512')
    lab_save_path = os.path.join(args.path, wave_band, 'lab_train_512')
    images = glob.glob(os.path.join(args.train_path, '*.tif'))
    labels = glob.glob(os.path.join(args.train_path, '*.png'))

    print('cutting band {} images'.format(wave_band))
    pic_cutting(args, images, img_save_path)
    print('cutting band {} labels'.format(wave_band))
    pic_cutting(args, labels, lab_save_path)
    pass

if __name__ == '__main__':
    main()
