"""
    author:nrzheng
    function:plot pie (计算文件夹内所有标签的类别占比，并保存pie图（单幅+总的）)
    edition:1.0
    date:2021.12.12

    you need to replace the "label_path, label_mapping"
"""

import os
from PIL import Image
import numpy as np
import collections
import matplotlib.pyplot as plt
from tqdm import tqdm
from pyecharts import Pie

Image.MAX_IMAGE_PIXELS = None
plt.rcParams['font.sans-serif']=['time new roman']

label_path = r'E:\try\aaa'
label_mapping = {'water':[0, 0, 255], 'baresoil':[139, 0, 0], 
                'road':[83, 134, 139], 'industry':[255, 0, 0], 
                'vegetation':[0, 255, 0], 'residential':[205, 173, 0], 
                'plantingarea':[139, 105, 20], 'other':[178, 34, 34], 
                'farms':[0, 139, 139], 'background':[0, 0, 0]}

def cal_label_count(label):
    """
        计算一张图的类别数量
    """
    h, w, c = label.shape
    label_count = collections.defaultdict()
    for i, (k, v) in enumerate(label_mapping.items()):
        tmp = np.array(label[:, :, :] == v, dtype='uint8')
        label_count[k] = (np.floor(tmp.sum(axis=2) / 3)).sum()
    return label_count

def cal_label_ratio(label_count):
    """
        计算类别占比
    """
    label_ratio = collections.defaultdict()
    for k, v in label_count.items():
        label_ratio[k] = v / sum(label_count.values())
    return label_ratio

def cal_all_label_count(all_label_count, label_count):
    """
        把每张图的类别都加一起
    """
    for k, v in label_count.items():
        if k in all_label_count:
            all_label_count[k] += v
        else:
            all_label_count[k] = v
    return all_label_count

def plot_pie(label_ratio, title=None, save_path=None):
    """
        画饼状图
    """
    plt.figure(figsize=(10, 10))    # 将画布设定为正方形，则绘制的饼图是正圆
    pie_label = []                  # 定义饼图的标签，标签是列表
    pie_value = []
    explode=[]                      # 设定各项距离圆心n个半径
    for k, v in label_ratio.items():
        pie_label.append(k)
        pie_value.append(v)
        explode.append(0.01)
    
    ######################
    # plt.pie(pie_value, explode=explode, labels=pie_label, autopct='%1.1f%%')
    patches, _, _ = plt.pie(pie_value, explode=explode, autopct='%1.1f%%')
    # patches, texts = plt.pie(pie_value, explode=explode)
    labels = ['{0} - {1:1.1f} %'.format(i, j * 100) for i, j in zip(pie_label, pie_value)]
    # patches, labels, dummy = zip(*sorted(zip(patches, labels, pie_value), key=lambda x: x[2], reverse=True))
    patches, labels, dummy = zip(*sorted(zip(patches, labels, pie_value), key=lambda x: x[2], reverse=True))
    # plt.legend(patches, labels, loc='upper left', bbox_to_anchor=(-0.1, 1.), fontsize=10)
    plt.legend(patches, labels, loc='upper left', bbox_to_anchor=(-0.1, 1.), fontsize=10)
    plt.title(title)
    plt.savefig(save_path)
    # plt.show()

def plot_pie_with_pyecharts(label_ratio, title=None, save_path=None):
    """
        用pyecharts的方法画饼状图
    """
    pie_label = []                  # 定义饼图的标签，标签是列表
    pie_value = []
    for k, v in label_ratio.items():
        pie_label.append(k)
        pie_value.append(v)
    # pie = Pie(title, title_pos='center', width=900, title_text_size=20)
    pie = Pie(title)
    pie.add('', pie_label, pie_value, is_label_show=True, legend_pos='right', legend_orient='vertical')
    pie.render(save_path)

def main():
    """
        主函数
    """
    all_label_count = collections.defaultdict()
    labels = os.listdir(label_path)
    for i in tqdm(labels, total=len(labels) + 1):
        label = Image.open(os.path.join(label_path, i))
        label = np.asarray(label)
        label_count = cal_label_count(label)
        label_ratio = cal_label_ratio(label_count)
        plot_pie(label_ratio, title='{}_pie'.format(i.split('.')[0]), save_path=os.path.join(label_path, '{}_pie.jpg'.format(i.split('.')[0])))
        # plot_pie_with_pyecharts(label_ratio, title='{}_pie'.format(i.split('.')[0]), save_path=os.path.join(label_path, '{}_pie.html'.format(i.split('.')[0])))

        all_label_count = cal_all_label_count(all_label_count, label_count)
        all_label_ratio = cal_label_ratio(all_label_count)

    plot_pie(all_label_ratio, title='all_label_pie', save_path=os.path.join(label_path, 'all_label_pie.jpg'))
    # plot_pie_with_pyecharts(all_label_ratio, title='all_label_pie', save_path=os.path.join(label_path, 'all_label_pie.html'))
    pass

if __name__ == '__main__':
    main()
