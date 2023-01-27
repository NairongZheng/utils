import os
import numpy as np
from tqdm import tqdm
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

LABEL_PATH = r'E:\try\new'
SAVE_PATH = r'E:\try\aaa'

def one2three(args, lab):
    # 可以用这个
    new_label = np.zeros([args.h, args.w, 3])
    for i, (k, v) in enumerate(new_rgb.items()):
        locals()['cls' + str(k)] = np.where(lab == k)
        new_label[eval('cls' + str(k))] = v
    return new_label

def main():
    labels = os.listdir(LABEL_PATH)
    for i in tqdm(labels, total=len(labels)):

        lab = Image.open(os.path.join(LABEL_PATH, i))
        width, height = lab.size
        new_label = np.zeros([height, width, 3])
        lab = np.asarray(lab)

        water = lab == 0
        baresoil = lab == 1
        road = lab == 2
        industry = lab == 3
        vegetation = lab == 4
        residential = lab == 5
        plantingarea = lab == 6
        other = lab == 7
        farms = lab == 8


        new_label[:, :, 0] = water * 0 + baresoil * 139 + road * 83 + industry * 255 + vegetation * 0 + residential * 205 + plantingarea * 139 + other * 178 + farms * 0
        new_label[:, :, 1] = water * 0 + baresoil * 0 + road * 134 + industry * 0 + vegetation * 255 + residential * 173 + plantingarea * 105 + other * 34 + farms * 139
        new_label[:, :, 2] = water * 255 + baresoil * 0 + road * 139 + industry * 0 + vegetation * 0 + residential * 0 + plantingarea * 20 + other * 34 + farms * 139


        aaa = Image.fromarray(np.uint8(new_label))
        aaa.save(os.path.join(SAVE_PATH, i))



if __name__ == '__main__':
    main()
