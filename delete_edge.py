"""
    author:nrzheng
    function:change edge to black
    edition:1.0
    data:2021.12.09

    you need to replace the "edge_path, images_path"
"""

import os
from PIL import Image
import numpy as np
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None

edge_path = r'D:\项目\data\dataset\raw\jiangsu_sheyang\sy_edge.png'
images_path = r'D:\项目\data\dataset\raw\jiangsu_sheyang'

def main():
    """
        主函数
    """
    edge = Image.open(edge_path)
    images = os.listdir(images_path)
    images.remove(os.path.split(edge_path)[1])
    for image in tqdm(images, total=len(images)):
        img = Image.open(os.path.join(images_path, image))
        img = np.asarray(img)
        h, w, c = img.shape
        new_img = np.zeros([h, w, c])
        new_edge = edge.resize((w, h))
        new_edge = np.asarray(new_edge)
        new_edge = new_edge == 255
        new_img[:, :, 0] = img[:, :, 0] * new_edge
        new_img[:, :, 1] = img[:, :, 1] * new_edge
        new_img[:, :, 2] = img[:, :, 2] * new_edge
        new_img = Image.fromarray(np.uint8(new_img))
        new_img.save(os.path.join(images_path, image))

    pass

if __name__ == '__main__':
    main()
