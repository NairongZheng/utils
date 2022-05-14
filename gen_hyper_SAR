"""
    Author:damonzheng
    function:generating hyper-SAR
    edition:1.0
    date:2022.5.14
"""

import argparse
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import gdal

Image.MAX_IMAGE_PIXELS = None

def parse_args():
    """
        参数设置
    """
    parser = argparse.ArgumentParser(description='generating hyper-SAR')
    parser.add_argument('--img_path', help='the path of all wave band file', default=r'E:\try\data\sheyang_3')
    parser.add_argument('--save_path', help='the path of save file', default=r'D:\项目\data\data_hyper_SAR\wanning_1')
    parser.add_argument('--bands', help='the selected bands(there must be S band)(,后不要带空格)', default='S,C,Ka')
    parser.add_argument('--output_ch', help='the number of channels of the output hyper-SAR', default=8)

    args = parser.parse_args()
    return args

def main():
    """
        主函数
    """
    args = parse_args()
    images = os.listdir(args.img_path)
    bands = args.bands.split(',')
    chose_images = {}
    for i in images:
        img_name = os.path.splitext(i)[0]
        band = img_name.split('_')[-1]
        if band in bands:
            chose_images[band] = i
    hyper_ch = len(chose_images) * 3
    S_image = Image.open(os.path.join(args.img_path, chose_images['S']))
    S_image = np.asarray(S_image)
    h, w = S_image.shape[:-1]
    del S_image
    output_hyper = np.zeros((h, w, hyper_ch))
    index = 0
    for i, (k, v) in tqdm(enumerate(chose_images.items()), total=len(chose_images)):
        use_image = Image.open(os.path.join(args.img_path, chose_images[k]))
        use_image = use_image.resize((w, h))
        use_image = np.asarray(use_image)
        output_hyper[:,:,index:index+3] = use_image
        del use_image
        index = index + 3
    # all_ch_data = output_hyper[:,:,:args.output_ch]
    # del output_hyper
    out_band_data = []
    for i in range(args.output_ch):
        out_band_data.append(output_hyper[:,:,i])
    del output_hyper

    file_name = os.path.join(args.save_path, img_name.split('_')[0] + '.tif')
    driver = gdal.GetDriverByName("GTiff")  # 创建文件驱动
    dataset = driver.Create(file_name, w, h, args.output_ch, 1)
    for i in range(args.output_ch):
        dataset.GetRasterBand(i + 1).WriteArray(out_band_data[i])

if __name__ == '__main__':
    main()
