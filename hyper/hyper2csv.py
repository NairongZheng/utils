"""
    author:damonzheng
    function:准备高光谱数据（图片分类/csv）
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

label_mapping = {0:[0, 0, 255], 1:[139, 0, 0], 2:[83, 134, 139], 
                3:[0, 255, 0], 4:[205, 173, 0], 5:[139, 105, 20], 6:[255, 0, 0]}


def parse_args():
    parser = argparse.ArgumentParser(description='preparing hyper-data')
    parser.add_argument('--data_path', help='the file path of hyper and label', default=r'E:\python_code\hyper_fusion\data\raw')
    parser.add_argument('--save_path', help='the save path', default=r'E:\python_code\hyper_fusion\data\save_path')
    parser.add_argument('--ratio', help='the ratio of data generation', default=0.1)
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


def lab321(lab):
    """
        三通道转单通道
    """
    tmp = lab.copy()
    lab_mask = np.zeros((lab.shape[0], lab.shape[1]))
    for i, (k, v) in enumerate(label_mapping.items()):
        lab_mask[(((tmp[:, :, 0] == v[0]) & (tmp[:, :, 1] == v[1])) & (tmp[:, :, 2] == v[2]))] = int(k)
    return lab_mask


def chose_points(args, lab_path):
    """
        选择要输出的像素点
    """
    lab = np.array(Image.open(lab_path))
    lab = lab321(lab)
    chose_point = np.zeros((lab.shape[0], lab.shape[1]))
    for i, (k, v) in enumerate(label_mapping.items()):
        cls_xy = np.where(lab == k)
        axis = np.array([[x, y] for x, y in zip(cls_xy[0], cls_xy[1])])    # [x, y]
        np.random.shuffle(axis)
        if len(axis) == 0:
            continue
        axis = axis[:int(len(axis) * args.ratio), :]
        chose_xy = (axis[:, 0], axis[:, 1])
        chose_point[chose_xy] = 1
    points_xy = np.where(chose_point == 1)
    cls = lab[points_xy]
    return points_xy[0], points_xy[1], cls


def save_csv(args, hyp_path, hwcls):
    img_name = os.path.split(hyp_path)[1].split('.')[0]
    img, height, width, band_num, geo, proj = read_img(hyp_path)
    img = hyper2numpy(img, height, width, band_num)
    # 设置列名
    column_name = np.array(['h', 'w'])
    band_name = []
    for i in range(img.shape[2]):
        band_name.append('band{}'.format(i + 1))
    column_name = np.hstack((column_name, band_name, 'sum', 'label'))
    # 创建空表
    df = pd.DataFrame()
    for h, w, cls in tqdm(hwcls, total=len(hwcls)):
        pos = np.array([h, w])
        pix = img[h, w, :]
        cls = np.array(cls)
        sum_ = np.sum(pix)
        each_item = np.hstack((pos, pix, sum_, cls))
        each_item = np.expand_dims(each_item, axis=0)
        each_item = pd.DataFrame(each_item)
        df = df.append(each_item, ignore_index=True)
    df.columns = column_name
    df.to_csv(os.path.join(args.save_path, img_name + '.csv'), index=False)
    pass


def main():
    args = parse_args()
    hyper = glob.glob(os.path.join(args.data_path, '*.tif'))
    label = glob.glob(os.path.join(args.data_path, '*.png'))
    hyper.sort()
    label.sort()
    hyper_and_label = [[x, y] for x, y in zip(hyper, label)]
    for hyp_path, lab_path in hyper_and_label:
        h, w, cls = chose_points(args, lab_path)
        hwcls = [[ii, jj, kk] for ii, jj, kk in zip(h, w, cls)]     # [h坐标, w坐标, 类别]
        save_csv(args, hyp_path, hwcls)
        pass
    pass


if __name__ == '__main__':
    main()
