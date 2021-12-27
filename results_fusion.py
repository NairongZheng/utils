"""
    author:nrzheng
    function:融合结果
    edition:1.0
    date:2021.12.27

    融合函数里面取哪个波段，还有优先级都是要改的
"""

import os
import argparse
from PIL import Image
import numpy as np
import glob
import cv2
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None

label_mapping = {0:[0,0,255], 1:[139,0,0], 2:[83,134,139], 
                3:[255,0,0], 4:[0,255,0], 5:[205,173,0], 
                6:[139,105,20], 7:[178,34,34], 8:[0,139,139]}
n_labels = len(label_mapping)

def get_cmap():
    labels = np.ndarray((n_labels, 3), dtype='uint8')
    for i, (k, v) in enumerate(label_mapping.items()):
        labels[i] = v
    cmap = np.zeros([768], dtype='uint8')
    index = 0
    for i in range(0, n_labels):
        for j in range(0, 3):
            cmap[index] = labels[i][j]
            index += 1
    print('cmap define finished')
    return cmap

def parse_args():
    """
        参数设置
    """
    parser = argparse.ArgumentParser(description='the fusion of all bands')
    parser.add_argument('--path', help='the path of all bands outputs', default=r'E:\python_code\yyjf\outputs\w18')
    parser.add_argument('--save_path', help='the path of the fusion results', default=r'E:\try\try')
    parser.add_argument('--bands', help='the used bands', default=None)
    parser.add_argument('--img_size', help='the size of final results', default=2048)

    args = parser.parse_args()
    return args

def img_resize(args, img, mode=cv2.INTER_LINEAR):
    """
        缩放图像并转成单通道
    """
    new_img = cv2.resize(img, (args.img_size, args.img_size), mode)
    temp = new_img[:,:,::-1].copy()
    new_mask = np.zeros((args.img_size, args.img_size))
    for i, (k, v) in enumerate(label_mapping.items()):
        new_mask[(((temp[:, :, 0] == v[0]) & (temp[:, :, 1] == v[1])) & (temp[:, :, 2] == v[2]))] = int(k)
    return new_mask

def fusion(new_mask, c_img, ka_img, s_img, sxz_img):
    """
        融合的函数
        water > plantingarea > farms > vegetation > industry > resident > baresoil > road > other
    """
    water = np.where(ka_img == 0)
    baresoil = np.where(ka_img == 1)
    road = np.where(ka_img == 2)
    industry = np.where(c_img == 3)
    vegetation = np.where(s_img == 4)
    resident = np.where(sxz_img == 5)
    plantingarea = np.where(ka_img == 6)
    other = np.where(sxz_img == 7)
    farms = np.where(s_img == 8)

    new_mask[other] = 7
    new_mask[road] = 2
    new_mask[baresoil] = 1
    new_mask[resident] = 5
    new_mask[industry] = 3
    new_mask[vegetation] = 4
    new_mask[plantingarea] = 6
    new_mask[water] = 0
    new_mask[farms] = 8

    return new_mask

def save_pred(preds, name, args):
    water = preds == 0
    baresoil = preds == 1
    road = preds == 2
    industry = preds == 3
    vegetation = preds == 4
    residential = preds == 5
    plantingarea = preds == 6
    other = preds == 7
    farms = preds == 8

    h = preds.shape[0]
    w = preds.shape[1]
    del preds
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :, 0] = water * 0 + baresoil * 139 + road * 83 + industry * 255 + vegetation * 0 + residential * 205 + plantingarea * 139 + other * 178 + farms * 0
    rgb[:, :, 1] = water * 0 + baresoil * 0 + road * 134 + industry * 0 + vegetation * 255 + residential * 173 + plantingarea * 105 + other * 34 + farms * 139
    rgb[:, :, 2] = water * 255 + baresoil * 0 + road * 139 + industry * 0 + vegetation * 0 + residential * 0 + plantingarea * 20 + other * 34 + farms * 139
    save_img = Image.fromarray(np.uint8(rgb))
    save_img.save(os.path.join(args.save_path, 'pred_fusion_' + name + '.png'))

def main():
    """
        主函数
    """
    args = parse_args()
    args.cmap = get_cmap()
    path = args.path
    c_images = glob.glob(os.path.join(path, 'C', 'pre', '*.png'))
    ka_images = glob.glob(os.path.join(path, 'Ka', 'pre', '*.png'))
    s_images = glob.glob(os.path.join(path, 'S', 'pre', '*.png'))
    sxz_images = glob.glob(os.path.join(path, 'sxz', 'pre', '*.png'))
    n_images = len(c_images)
    for i in tqdm(range(n_images)):
        c_img = cv2.imread(c_images[i])
        c_img = img_resize(args, c_img)
        ka_img = cv2.imread(ka_images[i])
        ka_img = img_resize(args, ka_img)
        s_img = cv2.imread(s_images[i])
        s_img = img_resize(args, s_img,)
        sxz_img = cv2.imread(sxz_images[i])
        sxz_img = img_resize(args, sxz_img)
        new_mask = fusion(ka_img, c_img, ka_img, s_img, sxz_img)        # 都融合到ka上去
        del(c_img)
        del(ka_img)
        del(s_img)
        del(sxz_img)
        name = str(i + 1).rjust(2, '0')
        save_pred(new_mask, name, args)

        pass
    pass

if __name__ == '__main__':
    main()
