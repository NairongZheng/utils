"""
    Author:nrzheng
    Function:calculate iou\miou\fwiou\oa
    Edition:3.0
    Date:2021.12.17

    you need to replace "LABEL_PATH, PRE_PATH, num_class, label_mapping" with your own value
    三通道
"""
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

LABEL_PATH = r'E:\try\try'
PRE_PATH = r'E:\python_code\yyjf\outputs\w18\S\pre'
num_class = 9
label_mapping = {0: [0, 0, 255], 1: [139, 0, 0], 2: [83, 134, 139], 
                3:[255, 0, 0], 4:[0, 255, 0], 5:[205, 173, 0], 
                6:[139, 105, 20], 7:[178, 34, 34], 8:[0, 139, 139], 
                255:[255, 255, 255]}

def _generate_matrix(gt_image, pre_image, num_class=7):
    """
        混淆矩阵:
        行: 实际,每一行之和表示该类别的真实样本数量; 
        列: 预测,每一列之和表示被预测为该类别的样本数量;
    """
    mask = (gt_image >= 0) & (gt_image < num_class)#ground truth中所有正确(值在[0, classe_num])的像素label的mask
    label = num_class * gt_image[mask].astype('int') + pre_image[mask] 
    # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
    count = np.bincount(label, minlength=num_class**2)
    confusion_matrix = count.reshape(num_class, num_class)#21 * 21(for pascal)
    return confusion_matrix

def _Class_IOU(confusion_matrix):
    MIoU = np.diag(confusion_matrix) / (
                np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
                np.diag(confusion_matrix))
    return MIoU

def meanIntersectionOverUnion(confusion_matrix):
    # Intersection = TP Union = TP + FP + FN
    # IoU = TP / (TP + FP + FN)
    intersection = np.diag(confusion_matrix)        # 交集: 混淆矩阵对角线
    union = np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - np.diag(confusion_matrix)     # 并集
    IoU = intersection / union
    mIoU = np.nanmean(IoU)
    # return mIoU
    return mIoU, IoU

# def Frequency_Weighted_Intersection_over_Union(confusion_matrix):
#     # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
#     freq = np.sum(confusion_matrix, axis=1) / np.sum(confusion_matrix)          # 真实样本分别占整张图的比例
#     iu = np.diag(confusion_matrix) / (
#             np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
#             np.diag(confusion_matrix))
#     FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()       # 有没有大于0其实无所谓,反正0乘了啥也是0,这么写是为了防止nan？
#     return FWIoU

def Frequency_Weighted_Intersection_over_Union(confusion_matrix):
    # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
    freq = np.sum(confusion_matrix, axis=1) / np.sum(confusion_matrix)          # 真实样本分别占整张图的比例
    iu = np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - np.diag(confusion_matrix) + 1e-7)
    FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()       # 有没有大于0其实无所谓,反正0乘了啥也是0,这么写是为了防止nan？
    return FWIoU

def change_channel(label):
    label_mask = np.zeros((label.shape[0], label.shape[1]))
    for k, v in label_mapping.items():
        label_mask[(((label[:, :, 0] == v[0]) & (label[:, :, 1] == v[1])) & (label[:, :, 2] == v[2]))] = int(k)
    return label_mask

def main():
    true_label_dir = os.listdir(LABEL_PATH)
    pre_label_dir = os.listdir(PRE_PATH)
    true_label_dir.sort()
    pre_label_dir.sort()

    confusion_matrix_all = np.zeros((num_class, num_class))
    for i in tqdm(range(0, len(true_label_dir)), total=len(true_label_dir)):
        true_label = Image.open(os.path.join(LABEL_PATH, true_label_dir[i]))
        true_label = np.asarray(true_label)
        true_label = change_channel(true_label)

        pre_label = Image.open(os.path.join(PRE_PATH, pre_label_dir[i]))
        pre_label = np.asarray(pre_label)
        pre_label = change_channel(pre_label)

        confusion_matrix = _generate_matrix(true_label.astype(np.int8), pre_label.astype(np.int8), num_class=num_class)
        confusion_matrix_all = confusion_matrix_all + confusion_matrix
        # miou = _Class_IOU(confusion_matrix_all)
        (miou, iou) = meanIntersectionOverUnion(confusion_matrix_all)
        fwiou = Frequency_Weighted_Intersection_over_Union(confusion_matrix_all)
        # acc = np.diag(confusion_matrix).sum() / confusion_matrix.sum()

    print('confusion_matrix_all is:\n', confusion_matrix_all)
    print('miou is:{}, and iou is{}:'.format(miou, iou))
    print('fwiou is:', fwiou)
    print('acc is:', np.diag(confusion_matrix_all).sum() / confusion_matrix_all.sum())

if __name__ == '__main__':
    main()
