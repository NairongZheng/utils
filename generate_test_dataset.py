"""
    author:nrzheng
    function:generate datasets(find test dataset)
    edition:1.0
    date:2021.12.13
"""

import os
import argparse
import glob
import collections
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None

def parse_args():
    """
        参数设置:
        海南万宁:
            points: np.array([[1398., 6158.], [2737., 10477.], [7173., 9418.], [12947., 12217.], [13426., 1279.], [4295., 3539.]])
            test_size: 2048
        江苏射阳:
            points: np.array([[1198., 7995.], [3620., 449.], [5268., 13217.], [2996., 16965.], [9412., 17864.], [3545., 3447.], [9886., 10343.]])
            test_size: 2048
        榆树:
            points: np.array([[6150., 1102.], [690., 2445.], [2718., 1306.]])
            test_size: 1024
    """
    parser = argparse.ArgumentParser(description='generate test dataset')
    parser.add_argument('--region', help='the name of region (wn, sy, ys)', default='ys')
    parser.add_argument('--image_and_laebl_path', help='the path of images and labels', default=r'D:\projects\data\dataset\raw\yushu')
    parser.add_argument('--save_path', help='the path of save', default=r'D:\projects\data\dataset\raw\yushu\dataset')

    # # 海南万宁
    # parser.add_argument('--points', help='the coordinates (x, y) of the upper left corner of the test area (Ka)',
    #                         default=np.array([[1398., 6158.], [2737., 10477.], [7173., 9418.], 
    #                                             [12947., 12217.], [13426., 1279.], [4295., 3539.]]))

    # # 江苏射阳
    # parser.add_argument('--points', help='the coordinates (x, y) of the upper left corner of the test area (Ka)',
    #                         default=np.array([[1198., 7995.], [3620., 449.], [5268., 13217.], [9886., 10343.], 
    #                                             [2996., 16965.], [9412., 17864.], [3545., 3447.]]))

    # 榆树
    parser.add_argument('--points', help='the coordinates (x, y) of the upper left corner of the test area (Ka)',
                            default=np.array([[6150., 1102.], [690., 2445.], [2718., 1306.]]))

    parser.add_argument('--test_size', help='the size of the test regions', default=1024)

    args = parser.parse_args()
    return args

def cal_points_region_size(args):
    """
        计算不同波段的test的位置和大小
    """
    wave_band = args.wave_band
    points = args.points
    test_size = args.test_size
    points_and_size = collections.defaultdict()
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

def plot_in_label(args, images_and_labels):
    """
        在标签中画出框的位置，可视化看看有没有取对
        images_and_labels每个里面都有4: 波段、图像路径、标签路径、点+尺寸
    """
    label = cv2.imread(images_and_labels[2])
    points = images_and_labels[3][0]
    size = images_and_labels[3][1]
    for point in points:
        upper_left = (point[0], point[1])
        lower_right = (point[0] + size, point[1] + size)
        cv2.rectangle(label, upper_left, lower_right, (255, 255, 255), 20)
    h, w, c = label.shape
    label = cv2.resize(label, (int(w / 2), int(h / 2)))
    # cv2.imshow('aaa', label)
    # cv2.waitKey(0)
    cv2.imwrite(os.path.join(args.save_path, '{}_thumb.jpg'.format(args.region)), label)

def crop_test(args, images_and_labels):
    """
        在原图和标签中把test区域截出来
        images_and_labels每个里面都有4: 波段、图像路径、标签路径、点+尺寸
    """
    wave_band = images_and_labels[0]
    image_path = images_and_labels[1]
    label_path = images_and_labels[2]
    points = images_and_labels[3][0]
    size = images_and_labels[3][1]

    # 对图像处理
    # image = cv2.imread(image_path)        # 图像太大的话没办法用cv2打开
    image1 = Image.open(image_path)
    image1 = np.asarray(image1)
    image = image1.copy()
    # image = image[:, :, ::-1]

    # 把每个区域截出来，并在原来的地方补成空白
    for i, point in enumerate(points):
        x = point[0]
        y = point[1]
        test_image = image[y:y + size, x:x + size, :]
        save_test_path = os.path.join(args.save_path, 'test', wave_band)
        if not os.path.exists(save_test_path):
            os.makedirs(save_test_path)
        # cv2.imwrite(os.path.join(save_test_path, '{}_{}_{}.tif'.format(args.region, wave_band, i + 1)), test_image)
        test_image = Image.fromarray(np.uint8(test_image))
        test_image.save(os.path.join(save_test_path, '{}_{}_{}.tif'.format(args.region, wave_band, i + 1)))
        image[y:y + size, x:x + size, :] = 255          # 截出来的区域补成空白
    
    save_train_path = os.path.join(args.save_path, 'train')
    if not os.path.exists(save_train_path):
        os.makedirs(save_train_path)
    image = Image.fromarray(np.uint8(image))
    image.save(os.path.join(save_train_path, '{}_{}.tif'.format(args.region, wave_band)))
    # cv2.imwrite(os.path.join(args.save_path, 'train', '{}_{}.tif'.format(args.region, wave_band)), image)

    # 对标签处理
    label = cv2.imread(label_path)
    for i, point in enumerate(points):
        x = point[0]
        y = point[1]
        test_label = label[y:y + size, x:x + size, :]
        save_test_path = os.path.join(args.save_path, 'test', wave_band)
        if not os.path.exists(save_test_path):
            os.makedirs(save_test_path)
        cv2.imwrite(os.path.join(save_test_path, '{}_{}_{}_label.png'.format(args.region, wave_band, i + 1)), test_label)
        label[y:y + size, x:x + size, :] = 255
    cv2.imwrite(os.path.join(args.save_path, 'train', '{}_{}_label.png'.format(args.region, wave_band)), label)

def main():
    """
        主函数
    """
    args = parse_args()
    images = glob.glob(os.path.join(args.image_and_laebl_path, '*.tif'))
    labels = glob.glob(os.path.join(args.image_and_laebl_path, '*.png'))
    images.sort()
    labels.sort()
    # images.sort(key=lambda x: os.path.split(x)[1].split('.')[0].split('_')[1])
    # labels.sort(key=lambda x: os.path.split(x)[1].split('.')[0].split('_')[1])
    args.wave_band = [os.path.split(i)[1].split('.')[0].split('_')[1] for i in images]
    points_and_size = cal_points_region_size(args)

    # images_and_labels每个里面都有4个: 波段、图像路径、标签路径、点+尺寸
    images_and_labels = [[y, i, j, k] for y, i, j, k in zip(args.wave_band, images, labels, points_and_size.values())]
    # plot_in_label(args, images_and_labels[0])         # 在切之前先框出来看看效果怎么样
    for i in tqdm(images_and_labels, total=len(images_and_labels)):
        if i[0] == 'Ka' and not args.region == 'ys':
            plot_in_label(args, i)
        elif i[0] == 'S' and args.region == 'ys':
            plot_in_label(args, i)
        crop_test(args, i)
    pass

if __name__ == '__main__':
    main()
