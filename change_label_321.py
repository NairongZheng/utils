import os
from PIL import Image
import numpy as np
from tqdm import tqdm

LABEL_PATH = r'E:\try\try'
SAVE_PATH = r'E:\try\new'
Image.MAX_IMAGE_PIXELS = None

def three2one(lab):
    # 可以用这个
    temp = lab.copy()
    label_mask = np.zeros((lab.shape[0], lab.shape[1]))
    for i, (k, v) in enumerate(row_rgb.items()):
        label_mask[(((temp[:, :, 0] == v[0]) & (temp[:, :, 1] == v[1])) & (temp[:, :, 2] == v[2]))] = int(k)
    return label_mask

def main():
    labels = os.listdir(LABEL_PATH)
    for label in tqdm(labels, total=len(labels)):
        lab = Image.open(os.path.join(LABEL_PATH, label))
        lab = np.asarray(lab)

        water = np.array((lab[:, :, :] == [0, 0, 255]), dtype='uint8')
        baresoil = np.array((lab[:, :, :] == [139, 0, 0]), dtype='uint8')
        road = np.array((lab[:, :, :] == [83, 134, 139]), dtype='uint8')
        industry = np.array((lab[:, :, :] == [255, 0, 0]), dtype='uint8')
        vegetation = np.array((lab[:, :, :] == [0, 255, 0]), dtype='uint8')
        residential = np.array((lab[:, :, :] == [205, 173, 0]), dtype='uint8')
        plantingarea = np.array((lab[:, :, :] == [139, 105, 20]), dtype='uint8')
        other = np.array((lab[:, :, :] == [178, 34, 34]), dtype='uint8')
        farms = np.array((lab[:, :, :] == [0, 139, 139]), dtype='uint8')

        water_ = np.floor(water.sum(axis=2) / 3) * 0
        baresoil_ = np.floor(baresoil.sum(axis=2) / 3) * 1
        road_ = np.floor(road.sum(axis=2) / 3) * 2
        industry_ = np.floor(industry.sum(axis=2) / 3) * 3
        vegetation_ = np.floor(vegetation.sum(axis=2) / 3) * 4
        residential_ = np.floor(residential.sum(axis=2) / 3) * 5
        plantingarea_ = np.floor(plantingarea.sum(axis=2) / 3) * 6
        other_ = np.floor(other.sum(axis=2) / 3) * 7
        farms_ = np.floor(farms.sum(axis=2) / 3) * 8

        lab = water_ + baresoil_ + road_ + industry_ + vegetation_ + residential_ + plantingarea_ + other_+ farms_

        aaa = Image.fromarray(np.uint8(lab))
        aaa.save(os.path.join(SAVE_PATH, label))


if __name__ == '__main__':
    main()
