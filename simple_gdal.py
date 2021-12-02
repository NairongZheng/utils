import os
from tqdm import tqdm
import numpy as np
from PIL import Image
import gdal

PIC_PATH = r'E:\data\xionganimg\XiongAn.img'
SIZE = 256
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

def main():
    all_band_data = []
    (img, height, width, band_num, geo, proj) = read_img(PIC_PATH)
    for i in range(0, band_num):           # 每个波段的数据存到数组
        all_band_data.append(img.GetRasterBand(i + 1))
    
if __name__ == '__main__':
    main()
