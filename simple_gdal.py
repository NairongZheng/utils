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

# 下面是一些片段，用来切图的
out_band_data = []
for band_data in all_band_data:
    out_band_data.append(band_data.ReadAsArray(col_start, row_start, SIZE, SIZE))

driver = gdal.GetDriverByName("GTiff")  # 数据类型必须有，要计算需要多大内存
small_pic_name = '绝对路径'

small_pic = driver.Create(small_pic_name, SIZE, SIZE, band_num, all_band_data[0].DataType)

# 根据反射变换参数计算新图的原点坐标
top_left_x = top_left_x + offset_x * w_e_pixel_resolution
top_left_y = top_left_y + offset_y * n_s_pixel_resolution

# 将计算后的值组装为一个元组，以方便设置
new_geo = (top_left_x, geo[1], geo[2], top_left_y, geo[4], geo[5])

small_pic.SetGeoTransform(new_geo)  # 写入仿射变换参数
small_pic.SetProjection(proj)  # 写入投影

for i in range(band_num):
    small_pic.GetRasterBand(i + 1).WriteArray(out_band_data[i])
del small_sar
