"""
    author:damonzheng
    function:高光谱归一化
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
    parser = argparse.ArgumentParser(description='hyper-data normalization')
    parser.add_argument('--data_path', help='the file path of hyper', default=r'E:\python_code\hyper_fusion\data\hyper_没归一化.tif')
    parser.add_argument('--save_path', help='the save path of hyper', default=r'E:\python_code\hyper_fusion\data\raw\hyper.tif')
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
        each_band = dataset.GetRasterBand(i + 1).ReadAsArray(0, 0, w, h)
        # img_mean = np.mean(each_band)
        # rate = 3
        img_new = each_band.copy()
        # img_new[each_band > (rate * img_mean + 1e-7)] = rate * img_mean
        img_new = np.uint8((img_new / np.max(img_new) + 1e-7) * 255.)
        all_band_data[:,:,i] = img_new
        pass
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


def main():
    args = parse_args()
    img, height, width, band_num, geo, proj = read_img(args.data_path)
    img = hyper2numpy(img, height, width, band_num)
    numpy2hyper_save(img, height, width, args.save_path)
    pass

if __name__ == '__main__':
    main()
