"""
    author:nrzheng
    function:find_edge
    edition:1.0
    data:2021.12.09

    you need to replace the "sxz_path, save_path"
"""

import os
from PIL import Image
import numpy as np

Image.MAX_IMAGE_PIXELS = None

sxz_path = r'D:\项目\data\海南万宁多维度SAR和可见光数据（25平方公里）\三线阵数据\sxz_wn_2021328_0.2m.tif'
save_path = r'D:\项目\data\海南万宁多维度SAR和可见光数据（25平方公里）\三线阵数据\edge.png'

def main():
    """
        主函数
    """
    image = Image.open(sxz_path)
    image = np.asarray(image)
    is_edge = np.array((image[:, :, :] == [255, 255, 255]), dtype='uint8')
    is_edge_ = np.floor(is_edge.sum(axis=2) / 3) * 1
    new_image = is_edge_ == 0
    new_image = new_image * 255
    znr = Image.fromarray(np.uint8(new_image))

    znr.save(save_path)
    pass

if __name__ == '__main__':
    main()
