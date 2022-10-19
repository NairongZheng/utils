"""
    author:damonzheng
    function:csv2iamges(classification)
    edition:1.0
    date:2022.10.19
"""
import argparse
import os
import numpy as np
from tqdm import tqdm
import cv2
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='preparing hyper-data')
    parser.add_argument('--data_path', help='the path of csv data', default=r'E:\python_code\hyper_fusion\data\save_path\hyper.csv')
    parser.add_argument('--save_path', help='the save path', default=r'E:\python_code\hyper_fusion\data\save_path\classes')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    df = pd.read_csv(args.data_path)
    df = np.array(df)
    for each_item in tqdm(df, total=len(df)):
        h = each_item[0]
        w = each_item[1]
        label = each_item[-1]
        band_val = each_item[2:-1]
        x = [i for i in range(len(band_val))]
        plt.plot(x, band_val)
        plt.axis('off')
        name = str(int(h)) + '_' + str(int(w)) + '.png'
        sav_path = os.path.join(args.save_path, str(int(label)))
        if not os.path.exists(sav_path):
            os.makedirs(sav_path)
        plt.savefig(os.path.join(sav_path, name))
        plt.close()
        pass
    pass


if __name__ == '__main__':
    main()
