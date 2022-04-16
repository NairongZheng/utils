"""
    author:nrzheng
    function:cut 1 picture to 4
    edition:1.0
    data:2024.04.11
    you need to replace the "image_path, save_path"
"""

import os
from PIL import Image
import numpy as np

Image.MAX_IMAGE_PIXELS = None

image_path = r'D:\项目\data\江苏射阳多维度SAR和可见光数据（25平方公里）\label\PixelLabelData\Label_1.png'
save_path = r'D:\项目\data\江苏射阳多维度SAR和可见光数据（25平方公里）\label\check\lab'
label_mapping = {'water':[0, 0, 255], 'baresoil':[139, 0, 0], 
                'road':[83, 134, 139], 'industry':[255, 0, 0], 
                'vegetation':[0, 255, 0], 'residential':[205, 173, 0], 
                'plantingarea':[139, 105, 20], 'other':[178, 34, 34], 
                'farms':[0, 139, 139]}

def main():
    """
        主函数
    """
    image = Image.open(image_path)
    image = np.asarray(image)
    h, w = image.shape
    image1 = image[:int(h/2), :int(w/2)]
    image2 = image[:int(h/2), int(w/2):]
    image3 = image[int(h/2):, :int(w/2)]
    image4 = image[int(h/2):, int(w/2):]

    image1 = Image.fromarray(np.uint8(image1))
    image1.save(os.path.join(save_path, '1.png'))
    image2 = Image.fromarray(np.uint8(image2))
    image2.save(os.path.join(save_path, '2.png'))
    image3 = Image.fromarray(np.uint8(image3))
    image3.save(os.path.join(save_path, '3.png'))
    image4 = Image.fromarray(np.uint8(image4))
    image4.save(os.path.join(save_path, '4.png'))
    pass

if __name__ == '__main__':
    main()

   
  
"""
    author:damonzheng
    function:connect 4 pic
    edition:1.0
    date:20220412
"""

import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None

def parse_args():
    """
        参数设置
    """
    parser = argparse.ArgumentParser(description='connecting pic')
    parser.add_argument('--img_path', help='the path of small images', default=r'E:\try\ccc')
    parser.add_argument('--save_path', help='the path of save images', default=r'E:\try\ccc')

    args = parser.parse_args()
    return args

def main():
    """
        主函数
    """
    args = parse_args()
    img_1 = Image.open(os.path.join(args.img_path, 'sy_sxz_1.jpg'))
    img_2 = Image.open(os.path.join(args.img_path, 'sy_sxz_2.jpg'))
    img_3 = Image.open(os.path.join(args.img_path, 'sy_sxz_3.jpg'))
    img_4 = Image.open(os.path.join(args.img_path, 'sy_sxz_4.jpg'))
    shape_1 = img_1.size
    shape_4 = img_4.size
    big_shape = (shape_1[1] + shape_4[1], shape_1[0] + shape_4[0])      # (h, w)
    to_image = np.zeros((big_shape[0], big_shape[1], 3), dtype=np.uint8)
    to_image[0:shape_1[1], 0:shape_1[0], :] = img_1
    to_image[0:shape_1[1]:, shape_1[0]:, :] = img_2
    to_image[shape_1[1]:, 0:shape_1[0], :] = img_3
    to_image[shape_1[1]:, shape_1[0]:, :] = img_4
    to_image = Image.fromarray(to_image)
    to_image.save(os.path.join(args.save_path, 'sy_sxz.jpg'))

if __name__ == '__main__':
    main()
