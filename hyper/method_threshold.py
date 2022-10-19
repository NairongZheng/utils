"""
    author:damonzheng
    function:method1--threshold
    edition:1.0
    date:2022.10.19
"""
import argparse
import gdal
from PIL import Image
import os
import glob
import numpy as np
from tqdm import tqdm
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description='preparing hyper-data')
    parser.add_argument('--data_path', help='the file path of hyper', default=r'E:\python_code\hyper_fusion\data\raw\hyper.tif')
    parser.add_argument('--save_path', help='the save path', default=r'E:\python_code\hyper_fusion\data\save_path\method1.png')
    args = parser.parse_args()
    return args


def read_img(img_path):
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


def main():
    args = parse_args()
    img, h, w, band_num, geo, proj = read_img(args.data_path)
    img = hyper2numpy(img, h, w, band_num)
    img_sum = np.sum(img, axis=2)
    pre = np.ones((h, w)) * 255

    water = np.where(img_sum < 100000)
    residence = np.where((400000 < img_sum) & (img_sum < 700000))
    road = np.where((700000 < img_sum) & (img_sum < 800000))
    meadow = np.where((250000 < img_sum) & (img_sum < 400000))
    baresoil = np.where((100000 < img_sum) & (img_sum < 250000))
    
    pre[water] = 0
    pre[residence] = 1
    pre[road] = 2
    pre[meadow] = 3
    pre[baresoil] = 4

    new_label = np.zeros([h, w, 3])

    water = pre == 0
    residence = pre == 1
    road = pre == 2
    meadow = pre == 3
    baresoil = pre == 4
    other = pre == 255


    new_label[:, :, 0] = water * 0 + residence * 139 + road * 83 + meadow * 0 + baresoil * 205 + other * 255
    new_label[:, :, 1] = water * 0 + residence * 0 + road * 134 + meadow * 255 + baresoil * 173 + other * 255
    new_label[:, :, 2] = water * 255 + residence * 0 + road * 139 + meadow * 0 + baresoil * 0 + other * 255
    aaa = Image.fromarray(np.uint8(new_label))
    aaa.save(args.save_path)


if __name__ == '__main__':
    main()
