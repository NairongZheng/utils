
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

path = r'E:\python_code\hyper_fusion\data\save_path\hyper.csv'
df = pd.read_csv(path)
df = df.iloc[:, -2:]
df_gp = df.groupby('label')
for label, lab_gp in df_gp:
    val = np.array(lab_gp)[:, 0]
    plt.hist(val, bins=12)
    # plt.show()
    plt.savefig(r'E:\python_code\hyper_fusion\data\save_path\aaa\{}.png'.format(str(int(label))))
    plt.close()

