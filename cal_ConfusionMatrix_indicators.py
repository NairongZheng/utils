"""
    Author:DamonZheng
    Function:计算混淆矩阵及语义分割的一些指标
    Edition:2.0
    Date:2021.10.11
    参考链接：https://www.cnblogs.com/vincent1997/p/10939747.html
"""

import numpy as np
import torch
from PIL import Image

label1_path = r'D:\competition\shengtengbei\data\train\labels\0.png'
label2_path = r'D:\competition\shengtengbei\data\train\labels\1.png'

def get_confusion_matrix_1(seg_gt, seg_pred, num_class, ignore=-1):
    """
    Calcute the confusion matrix by given label and pred
    """
    with torch.no_grad():
        # output = pred.cpu().numpy().transpose(0, 2, 3, 1)
        # # output = pred.transpose(0, 2, 3, 1)

        # seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
        # seg_gt = np.asarray(
        #     label.cpu().numpy()[:, :size[-2], :size[-1]], dtype=np.int)

        ignore_index = seg_gt != ignore     # 找到不是ignore是所有标签（布尔值矩阵(512, 512)（一堆true跟false））

        seg_gt = seg_gt[ignore_index]       # 所有true位置的值
        seg_pred = seg_pred[ignore_index]

        # 以行优先的方式用一维向量来存储二维信息!!!!!!!!!!!!
        # seg_gt[0]=1,seg_pred[0]=3有index[0]=1*num_class+3=22
        # index[0]=22表示第0个像素点本属于第1类的却被误判为3类，于是confusion_matrix[1][3]计数加一
        index = (seg_gt * num_class + seg_pred).astype('int32')
        label_count = np.bincount(index)
        confusion_matrix = np.zeros((num_class, num_class))

        for i_label in range(num_class):        # 0, 1, 2, ..., 8
            for i_pred in range(num_class):     # 0, 1, 2, ..., 8
                # 0*8+0, 0*8+1, ..., 8*8+8，每一次对应一种判断情况
                cur_index = i_label * num_class + i_pred
                if cur_index < len(label_count):
                    confusion_matrix[i_label,
                                     i_pred] = label_count[cur_index]   # 矩阵放入对应判断情况的次数
        return confusion_matrix

def get_confusion_matrix_2(gt_image, pre_image, num_class=7):
    
    # ground truth中所有正确(值在[0, classe_num])的像素label的mask
    # 就像上面那个函数中的ignore_index
    mask = (gt_image >= 0) & (gt_image < num_class)

    label = num_class * gt_image[mask].astype('int') + pre_image[mask]  # 就像上面的index
    # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
    count = np.bincount(label, minlength=num_class**2)                  # 就像上面的label_count，不过设置了参数minlength
    confusion_matrix = count.reshape(num_class, num_class)
    return confusion_matrix

def main():
    label1 = Image.open(label1_path)
    label1 = np.asarray(label1)
    label2 = Image.open(label2_path)
    label2 = np.asarray(label2)

    confusion_matrix1 = get_confusion_matrix_1(label1, label2, 9, 255)
    confusion_matrix2 = get_confusion_matrix_2(label1, label2, 9)

    confusion_matrix = confusion_matrix1

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)      # 混淆矩阵的对角线

    # oa
    oa = np.sum(tp) / (np.sum(confusion_matrix) + 1e-7)

    # iou and miou
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()

    # fwiou
    freq = pos / (np.sum(confusion_matrix) + 1e-7)
    iu = tp / (pos + res - tp + 1e-7)
    FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()

    print('iou:{}, miou:{}, fwiou:{}, oa:{}'.format(IoU_array, mean_IoU, FWIoU, oa))
    pass

if __name__ == '__main__':
    main()
# miou:0.0062210302237937255, fwiou:0.0206850898624856, oa:0.03412628173826823
# miou:0.0062210302237937255, fwiou:0.0206850898624856, oa:0.03412628173826823
