"""
    author:damonzheng
    function:遍历文件夹中的tif图像，统计每个通道的直方图
    date:2021.12.3

    you need to change "img_dir, save_dir, img_ext"
"""
import glob
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

#解决中文显示问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

Image.MAX_IMAGE_PIXELS = None

img_dir = r'D:\项目\data\榆树三线阵及SAR数据（25平方公里）'
save_dir = r'D:\项目\data\榆树三线阵及SAR数据（25平方公里）\直方图'
img_ext = '.tif'

def plot_oneimage_hist(path):
    """
        绘制一张图像的直方图
        入参是绝对路径
    """
    pic_name = os.path.split(path)[1].split('.')[0]

    img = Image.open(path)
    img = np.asarray(img)
    h, w, c = img.shape

    colors = ('r', 'g', 'b')
    for i, col in enumerate(colors):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        if i == 0:
            hist_all = hist
        else:
            hist_all += hist
        plt.plot(hist, color=col, label='{}'.format(col))
        plt.xlim([0, 256])
    plt.plot(hist_all, color='black', label='all_channels')
    plt.title('江苏射阳{}'.format(pic_name), color='black', loc='center')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(save_dir, pic_name + '.png'))
    plt.close()
    # plt.show()
    pass

def main():
    """
        主函数
    """
    all_img = glob.glob(os.path.join(img_dir, '*{}'.format(img_ext)))
    for img in tqdm(all_img, total=len(all_img)):
        plot_oneimage_hist(img)

if __name__ == '__main__':
    main()

