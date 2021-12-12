"""
    author:nrzheng
    function:mask + image
    edition:1.0
    date:2021.12.12
"""

import os
from PIL import Image
import cv2

image_path = r'E:\try\try\ya_S.tif'
label_path = r'E:\try\try\ys_label.png'
save_path = r'E:\try\try\ys_S_mask.jpg'

def main():
    """
        主函数
    """
    image = cv2.imread(image_path)
    label = cv2.imread(label_path)
    combine = cv2.addWeighted(image, 0.7, label, 0.3, 0)
    cv2.imwrite(save_path, combine)
    pass


if __name__ == '__main__':
    main()
