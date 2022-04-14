"""
    author:damonzheng
    function:generating public dataset
    edition:1.0
    date:2022.04.14
    在运行的时候会有内存限制...
"""

import argparse
import os
from PIL import Image
import numpy as np
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None

def parse_arge():
    """
        参数设置
    """
    parser = argparse.ArgumentParser(description='generating public dataset')
    parser.add_argument('--img_path', help='the path of all band SAR and one label', default=r'D:\code_python\try\bbb')
    parser.add_argument('--save_path', help='the path of saving img', default=r'D:\code_python\try\output')
    parser.add_argument('--img_size', help='the size of C image', default=1024)
    args = parser.parse_args()
    return args

def gen_file(args, band):
    """
        创建保存文件夹, 每个波段创建一个
    """
    img_save_path = os.path.join(args.save_path, band)
    if not os.path.exists(img_save_path):               # 判断是否有保存的文件夹, 没有则创建
        os.makedirs(img_save_path)
    for file in os.listdir(img_save_path):              # 清空文件夹
        os.remove(os.path.join(img_save_path, file))
    return img_save_path

def cal_small_img_size(args, band):
    """
        计算不同波段切图的大小
        不同波段分辨率如下:{'C':0.5, 'X':0.5, 'Ka':0.3, 'L':1, 'P':1, 'S':1, 'sxz':0.2, 'label':0.3}
        以C波段设置的小图大小为基准!!!!!!!!!!!!!!!!!!!!!
        如果C设置为1024, 那么其他各波段的大小为:
        {'C':1024, 'X':1024, 'Ka':1707, 'L':512, 'P':512, 'S':512, 'sxz':2560, 'label':1707}
    """
    resolution = 0.5
    if band == 'X' or band == 'C':
        rate = resolution / 0.5
    elif band == 'Ka' or band == 'label':
        rate = resolution / 0.3
    elif band == 'L' or band == 'P' or band == 'S':
        rate = resolution / 1
    elif band == 'sxz':
        rate = resolution / 0.2
    small_img_size = round(args.img_size * rate)
    return small_img_size

def gen_pad_img(raw_img_shape, small_img_size):
    """
        生成需要padding后的图像, 都是0, 出去该函数再赋值
    """
    h, w, c = raw_img_shape
    row = int(h / small_img_size) + 1
    col = int(w / small_img_size) + 1
    pad_img = np.zeros((row * small_img_size, col * small_img_size, c))
    return row, col, pad_img

def cutting_img(row, col, small_img_size, pad_img, region_name, region_num, band, ext, save_path):
    """
        切图, 没有重叠的切!!!!!!!!
    """
    stride = small_img_size
    id_hang = 1
    for i in tqdm(range(row), total=row):
        row_start = i * stride
        row_end = i * stride + stride
        id_lie = 1
        for j in range(col):
            col_start = j * stride
            col_end = j * stride + stride
            small_pic = pad_img[row_start:row_end, col_start:col_end, :]
            small_name = region_name + '_' + region_num + '_' + band + '_' + str(id_hang).rjust(3, '0') + '_' + str(id_lie).rjust(3, '0') + '_' + ext
            small_pic = Image.fromarray(np.uint8(small_pic))
            small_pic.save(os.path.join(save_path, small_name))
            id_lie += 1
        id_hang += 1

def main(args):
    """
        主函数
    """
    images = os.listdir(args.img_path)
    for image in images:
        img_name, extention = os.path.splitext(image)           # [图像名, 扩展名]
        region_name, region_num, band = img_name.split('_')     # [区域名, 区域编号, 波段]
        save_path = gen_file(args, band)                        # 根据波段创建保存的文件夹
        small_img_size = cal_small_img_size(args, band)         # 计算不同波段切出来的小图尺寸是多少
        big_img = Image.open(os.path.join(args.img_path, image))        # 打开待切大图
        big_img = np.asarray(big_img)
        raw_img_shape = big_img.shape                                   # 待切大图的宽, 高, 通道
        row, col, pad_img = gen_pad_img(raw_img_shape, small_img_size)  # 计算行, 列, 并返回padding后的图像尺寸
        pad_img[:raw_img_shape[0], :raw_img_shape[1], :] = big_img
        del big_img
        cutting_img(row, col, small_img_size, pad_img, region_name, region_num, band, extention, save_path)

if __name__ == '__main__':
    args = parse_arge()
    main(args)
