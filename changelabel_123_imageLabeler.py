"""
    author:nrzheng
    function:change the channel of labels from 3 to 1（matlab标完出来转的，所以一张一张转的，没写循环）
    edition:xxx
    data:2021.12.09

    you need to replace the "label_path, save_path, label_mapping"
"""

import os
from PIL import Image
import numpy as np

Image.MAX_IMAGE_PIXELS = None

label_path = r'D:\项目\data\江苏射阳多维度SAR和可见光数据（25平方公里）\label\PixelLabelData_4\Label_1.png'
save_path = r'D:\项目\data\江苏射阳多维度SAR和可见光数据（25平方公里）\label\PixelLabelData_4\label.png'
label_mapping = {'water':[0, 0, 255], 'baresoil':[139, 0, 0], 
                'road':[83, 134, 139], 'industry':[255, 0, 0], 
                'vegetation':[0, 255, 0], 'residential':[205, 173, 0], 
                'plantingarea':[139, 105, 20], 'other':[178, 34, 34], 
                'farms':[0, 139, 139]}

def main():
    """
        主函数
    """
    label = Image.open(label_path)
    label = np.asarray(label)
    h, w = label.shape
    new_label = np.zeros([h, w, 3])

    # 法一：速度比较慢
    for i, (k, v) in enumerate(label_mapping.items()):
        globals()[k] = label == i + 1                       # 项目的数据是白边，标了之后自动变成0，所以要+1
    for i, (k, v) in enumerate(label_mapping.items()):
        new_label[:, :, 0] += globals()[k] * v[0]
        new_label[:, :, 1] += globals()[k] * v[1]
        new_label[:, :, 2] += globals()[k] * v[2]

    # 法二：速度比较快，但是改起来麻烦
    # background = label == 0                               # 项目的数据是s白边，标了之后自动变成0，所以要加上
    # water = label == 1
    # baresoil = label == 2
    # road = label == 3
    # industry = label == 4
    # vegetation = label == 5
    # residential = label == 6
    # plantingarea = label == 7
    # other = label == 8
    # farms = label == 9

    # new_label[:, :, 0] = water * 0 + baresoil * 139 + road * 83 + industry * 255 + vegetation * 0 + residential * 205 + plantingarea * 139 + other * 178 + farms * 0
    # new_label[:, :, 1] = water * 0 + baresoil * 0 + road * 134 + industry * 0 + vegetation * 255 + residential * 173 + plantingarea * 105 + other * 34 + farms * 139
    # new_label[:, :, 2] = water * 255 + baresoil * 0 + road * 139 + industry * 0 + vegetation * 0 + residential * 0 + plantingarea * 20 + other * 34 + farms * 139

    znr = Image.fromarray(np.uint8(new_label))
    znr.save(save_path)
    pass

if __name__ == '__main__':
    main()
