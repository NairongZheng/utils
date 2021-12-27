"""
    Author:DamonZheng
    Function:connecting pictures
    Edition:final
    Date:2021.12.26
"""

import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None

label_mapping = {0:[0,0,255], 1:[139,0,0], 2:[83,134,139], 
                3:[255,0,0], 4:[205,173,0], 5:[0,255,0], 
                6:[0,139,0], 7:[189,183,107], 8:[178,34,34]}
n_labels = len(label_mapping)

def parse_args():
    """
        参数设置
    """
    parser = argparse.ArgumentParser(description='connecting pic')
    parser.add_argument('--path', help='the path of small images', default=r'F:\znr_GF3\dataset\img_test_256')
    parser.add_argument('--save_path', help='the path of save images', default=r'F:\znr_GF3\dataset')
    parser.add_argument('--ext', help='the extention of big images', default='.tif')
    parser.add_argument('--channels', help='the number of channels', default=1)
    parser.add_argument('--if_cmap', help='if using the color map or not', default=False)

    args = parser.parse_args()
    return args

def get_cmap():
    labels = np.ndarray((n_labels, 3), dtype='uint8')
    for i, (k, v) in enumerate(label_mapping.items()):
        labels[i] = v
    cmap = np.zeros([768], dtype='uint8')
    index = 0
    for i in range(0, n_labels):
        for j in range(0, 3):
            cmap[index] = labels[i][j]
            index += 1
    print('cmap define finished')
    return cmap

def img_con(args):
    small_img_path = args.path
    img_name = os.listdir(small_img_path)
    img_name.sort()
    
    small_pic_num = len(img_name)
    big_pic_num = int(img_name[small_pic_num - 1].split('image')[1].split('_')[0])

    # 创建一个列表。每个元素存放同一个大图切下来的所有小图名
    znr = [[] for i in range(0, big_pic_num)]
    for i in range(0, small_pic_num):
        znr[int(img_name[i].split('image')[1].split('_')[0]) - 1].append(img_name[i])
    
    for i in range(0, big_pic_num):
        k = 0
        small_pic_num_2 = len(znr[i])
        small_size = Image.open(os.path.join(small_img_path, znr[i][0])).size[0]
        row = int(znr[i][small_pic_num_2 - 1].split('row_')[0].split('_')[1])
        col = int(znr[i][small_pic_num_2 - 1].split('row_')[1].split('col')[0])
        if args.channels == 3:
            to_image = np.zeros((row * small_size, col * small_size, 3), dtype=np.uint8)
        elif args.channels == 1:
            to_image = np.zeros((row * small_size, col * small_size), dtype=np.uint8)

        for y in tqdm(range(0, row), total=row):
            row_start = y * small_size
            row_end = row_start + small_size
            for x in range(0, col):
                col_start = x * small_size
                col_end = col_start + small_size
                small_pic = Image.open(os.path.join(small_img_path, znr[i][k]))
                k += 1
                if args.channels == 3:
                    to_image[row_start:row_end, col_start:col_end, :] = small_pic
                elif args.channels == 1:
                    to_image[row_start:row_end, col_start:col_end] = small_pic
        to_image = Image.fromarray(to_image)
        if args.if_cmap:
            to_image.putpalette(args.cmap)
        save_img_name = '{}{}'.format(i + 1, args.ext)
        to_image.save(os.path.join(args.save_path, save_img_name))

def main():
    """
        主函数
    """
    args = parse_arge()
    args.cmap = get_cmap()
    img_con(args)

if __name__ == '__main__':
    main()
