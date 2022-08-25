

import os
from tqdm import tqdm
import numpy as np
from PIL import Image
import gdal

PIC_PATH = r'E:\try\ddd\C-SAR.tif'
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

    # 读取所有数据
    (img, height, width, band_num, geo, proj) = read_img(PIC_PATH)

    # 每个波段的数据存到数组
    for i in range(0, band_num):
        all_band_data.append(img.GetRasterBand(i + 1))
    
    # 输出的部分
    out_band_data = []
    for band_data in all_band_data:
        out_band_data.append(band_data.ReadAsArray(0, 0, 4000, 4000))       # (col_start, row_start, width, height)
    driver = gdal.GetDriverByName("GTiff")  # 数据类型必须有，要计算需要多大内存
    small_pic_name = r'E:\try\ddd\C-SAR-part.tif'
    small_pic = driver.Create(small_pic_name, 4000, 4000, band_num, all_band_data[0].DataType)
    small_pic.SetGeoTransform(geo)  # 写入仿射变换参数
    small_pic.SetProjection(proj)  # 写入投影
    for i in range(band_num):
        small_pic.GetRasterBand(i + 1).WriteArray(out_band_data[i])
    del small_pic
    pass

if __name__ == '__main__':
    main()
