"""
    Author:damonzheng
    function:generating the dataset of hyper-SAR(find test dataset)
    edition:1.0
    date:2022.5.15
"""

import os
import argparse
import glob
import numpy as np
import gdal
from PIL import Image

def parse_args():
    """
        参数设置:
        海南万宁:
            points: np.array([[1398., 6158.], [2737., 10477.], [7173., 9418.], [12947., 12217.], [13426., 1279.], [4295., 3539.]])
            test_size: 2048
        江苏射阳:
            points: np.array([[1198., 7995.], [3620., 449.], [5268., 13217.], [2996., 16965.], [9412., 17864.], [3545., 3447.], [9886., 10343.]])
            test_size: 2048
    """
    parser = argparse.ArgumentParser(description='generate the test dataset of hyper-SAR')
    parser.add_argument('--region', help='the name of region (wn, sy)', default='wn')
    parser.add_argument('--wave_band', help='the wave band(hyper-SAR都用的S, 这里不用改)', default='S')
    parser.add_argument('--image_and_laebl_path', help='the path of images and labels', default=r'D:\work\项目\data\data_hyper_SAR\raw_data\wanning')
    parser.add_argument('--save_path', help='the path of save', default=r'D:\work\项目\data\data_hyper_SAR\raw_data\wanning\dataset')

    # 海南万宁
    parser.add_argument('--points', help='the coordinates (x, y) of the upper left corner of the test area (Ka)',
                            default=np.array([[1398., 6158.], [2737., 10477.], [7173., 9418.], 
                                                [12947., 12217.], [13426., 1279.], [4295., 3539.]]))

    # # 江苏射阳
    # parser.add_argument('--points', help='the coordinates (x, y) of the upper left corner of the test area (Ka)',
    #                         default=np.array([[1198., 7995.], [3620., 449.], [5268., 13217.], [9886., 10343.], 
    #                                             [2996., 16965.], [9412., 17864.], [3545., 3447.]]))

    parser.add_argument('--test_size', help='the size of the test regions', default=2048)

    args = parser.parse_args()
    return args

def cal_points_region_size(args):
    """
        计算不同波段的test的位置和大小
    """
    wave_band = args.wave_band
    points = args.points
    test_size = args.test_size
    points_and_size = {}
    if args.region == 'ys':
        resolution = 1
    else:
        resolution = 0.3
    for i in wave_band:
        points_and_size[i] = []
    for k, v in points_and_size.items():
        if k == 'sxz':
            rate = resolution / 0.2
        elif k == 'C' or k == 'X':
            rate = resolution / 0.5
        elif k == 'L' or k== 'P' or k == 'S':
            rate = resolution / 1
        elif k == 'Ka':
            rate = resolution / 0.3

        v.append(np.around(points * rate).astype(dtype=int).tolist())
        v.append(round(test_size * rate))

    return points_and_size

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

def crop_test(args, images_and_labels):
    """
        在原图和标签中把test区域截出来
        images_and_labels每个里面都有3: 图像路径、标签路径、点+尺寸
    """
    image_path = images_and_labels[0]
    label_path = images_and_labels[1]
    points = images_and_labels[2][0]
    size = images_and_labels[2][1]

    ## 对图像处理
    (img, height, width, band_num, geo, proj) = read_hyper(image_path)
    img = hyper2numpy(img, height, width, band_num)         # 把gdal读出来的格式转成numpy多通道的格式

    # 把每个区域截出来，并在原来的地方补成空白
    for i, point in enumerate(points):
        x = point[0]
        y = point[1]
        test_image = img[y:y + size, x:x + size, :]     # 选出test
        save_test_path = os.path.join(args.save_path, 'test')
        if not os.path.exists(save_test_path):
            os.makedirs(save_test_path)

        # 保存截取出来的test
        test_save_filename = os.path.join(save_test_path, '{}_{}.tif'.format(args.region, i + 1))
        numpy2hyper_save(test_image, size, size, test_save_filename)

        img[y:y + size, x:x + size, :] = 255            # 截出来的区域补成空白

    # 保存截出来test后的train
    save_train_path = os.path.join(args.save_path, 'train')
    if not os.path.exists(save_train_path):
        os.makedirs(save_train_path)
    train_save_filename = os.path.join(save_train_path, '{}.tif'.format(args.region))
    numpy2hyper_save(img, height, width, train_save_filename)
    del img
    

    ## 对标签处理
    label1 = Image.open(label_path)
    label1 = np.asarray(label1)
    label = label1.copy()
    for i, point in enumerate(points):
        x = point[0]
        y = point[1]
        test_label = label[y:y + size, x:x + size, :]
        test_label = Image.fromarray(np.uint8(test_label))
        test_label.save(os.path.join(save_test_path, '{}_{}_label.png'.format(args.region, i + 1)))
        label[y:y + size, x:x + size, :] = 255
    label = Image.fromarray(np.uint8(label))
    label.save(os.path.join(args.save_path, 'train', '{}_label.png'.format(args.region)))

def main():
    """
        主函数
    """
    args = parse_args()
    images = glob.glob(os.path.join(args.image_and_laebl_path, '*.tif'))
    labels = glob.glob(os.path.join(args.image_and_laebl_path, '*.png'))
    points_and_size = cal_points_region_size(args)
    images_and_labels = [[i, j, k] for i, j, k in zip(images, labels, points_and_size.values())]
    for i in images_and_labels:
        crop_test(args, i)

if __name__ == '__main__':
    main()
