"""
    Author:damonzheng
    Function:cutting the hyper-SAR
    Edition:1.0
    Date:2022.5.15
"""

import os
import argparse
import numpy as np
import glob
from tqdm import tqdm
import gdal
import math

def parse_args():
    parser = argparse.ArgumentParser(description='cutting the hyper-SAR')
    parser.add_argument('--img_path', help='the path of images', default=r'D:\项目\data\data_hyper_SAR\dataset\train')
    parser.add_argument('--save_path', help='the path of save file', default=r'D:\项目\data\data_hyper_SAR\dataset\img_train_256')
    parser.add_argument('--ext', help='the extension of the images', default='.tif')
    parser.add_argument('--size', help='the cutting size', default=256)
    parser.add_argument('--strides', help='the moving stride', default=256)
    args = parser.parse_args()
    return args

def read_hyper(img_path):
    """
        读取图像信息
    """
    img = gdal.Open(img_path)
    height = img.RasterYSize        # 获取图像的行数
    width = img.RasterXSize         # 获取图像的列数
    band_num = img.RasterCount      # 获取图像波段数

    geo = img.GetGeoTransform()     # 仿射矩阵
    proj = img.GetProjection()      # 地图投影信息，字符串表示

    return img, height, width, band_num, geo, proj

def hyper2numpy(dataset, h, w, band_num):
    """
        把gdal读出来的hyper的dataset格式转成矩阵形式
    """
    all_band_data = np.zeros((h, w, band_num))
    for i in range(0, band_num):
        all_band_data[:,:,i] = dataset.GetRasterBand(i + 1).ReadAsArray(0, 0, w, h)
    return all_band_data

def numpy2hyper_save(array, h, w, save_path):
    """
        把多通道numpy保存成hyper
    """
    out_band_data = []
    band_num = array.shape[2]
    for i in range(band_num):
        out_band_data.append(array[:,:,i])
    del array
    driver = gdal.GetDriverByName("GTiff")  # 创建文件驱动
    dataset = driver.Create(save_path, w, h, band_num, 1)
    for i in range(band_num):
        dataset.GetRasterBand(i + 1).WriteArray(out_band_data[i])

def cutting_img(args, images_path):
    """
        切图
    """
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for ii, image_path in enumerate(images_path):
        image_names = os.path.split(image_path)[1]
        image_name, ext = os.path.splitext(image_names)
        (img, height, width, band_num, geo, proj) = read_hyper(image_path)
        img = hyper2numpy(img, height, width, band_num)         # 把gdal读出来的格式转成numpy多通道的格式

        stride = args.strides
        cut_h = args.size
        cut_w = args.size

        # 计算切图的行数和列数
        row = math.floor((height - cut_h) / stride)
        column = math.floor((width - cut_w) / stride)

        id_hang = 1
        for i in tqdm(range(row), total=row):
            row_start = i * stride
            row_end = i * stride + cut_h
            id_lie = 1
            for j in range(column):
                col_start = j * stride
                col_end = j * stride + cut_w
                small_pic = img[row_start:row_end, col_start:col_end, :]
                small_name = 'image' + str(ii + 1).rjust(3, '0') + '_' + str(id_hang).rjust(3, '0') + 'row_' + str(id_lie).rjust(3, '0') + 'col' + ext
                numpy2hyper_save(small_pic, cut_h, cut_w, os.path.join(save_path, small_name))
                id_lie += 1
            id_hang += 1


def main():
    """
        主函数
    """
    args = parse_args()
    images_path = glob.glob(os.path.join(args.img_path, '*{}'.format(args.ext)))
    cutting_img(args, images_path)

if __name__ == '__main__':
    main()
