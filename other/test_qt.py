"""
    author:damonzheng
    function:deeplabv3p_pytorch_test
    edition:1.0
    date:2021.11.21
"""
from PySide2.QtWidgets import QApplication, QMainWindow, QPushButton,  QPlainTextEdit, QTextEdit, QMessageBox
import argparse
import cv2
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import math
import time

import hrnet

from default import _C as config
from default import update_config

import torch
from torch.nn import functional as F
import torch.backends.cudnn as cudnn


os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def get_cmap(n_labels, label_dic):
    labels = np.ndarray((n_labels, 3), dtype='uint8')
    for i , (k, v) in enumerate(label_dic.items()):
        labels[i] = v
    cmap = np.zeros([768], dtype='uint8')
    index = 0
    for i in range(0, n_labels):
        for j in range(0, 3):
            cmap[index] = labels[i][j]
            index += 1
    return cmap

class TestDataLoader():
    """
        test data loader
    """
    def __init__(self, image_path):
        self.image_path = image_path
        self.crop_size = (256, 256)
        self.num_classes = 9
        self.samples, self.n_row, self.n_col, self.Height, self.Width = self.make_dataset(self.image_path)

    def make_dataset(self, img_path, height=256, width=256, stride=256):
        # img = Image.open(img_path)
        # img = np.array(img, dtype='float32')
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = img.astype(np.float32)[:, :, ::-1]

        Height = img.shape[0]
        Width = img.shape[1]
        # ch = img.shape[2]

        if (Height % height == 0) and (Width % width == 0):
            print('nice image size for slice')
        else:
            pad_h = height - (Height % height)
            pad_w = width - (Width % width)
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), 'constant')

        Height2 = img.shape[0]
        Width2 = img.shape[1]
        if (Height2 % height == 0) and (Width2 % width == 0):
            print('nice padding image size for slice')

        n_row = math.floor((Height2 - height) / stride) + 1
        n_col = math.floor((Width2 - width) / stride) + 1

        samples = np.zeros((n_row * n_col, height, width, 3), dtype=np.uint8)

        K = 0
        for m in range(n_row):
            row_start = m * stride
            row_end = m * stride + height
            for n in range(n_col):
                col_start = n * stride
                col_end = n * stride + width
                img_mn = img[row_start:row_end, col_start:col_end]
                samples[K, :, :, :] = img_mn
                K += 1

        return samples.copy(), n_row, n_col, Height, Width
    
    def input_transform(self, image):
        image = image.astype(np.float32)
        image = image / 127.5 -1
        return image

    def gen_sample(self, image):
        image = self.input_transform(image)
        image = image.transpose((2, 0, 1))
        return image
    
    def __getitem__(self, index):
        sample = self.samples[index]                # [h, w, c], RGB
        sample = self.gen_sample(sample)

        return sample.copy(), self.n_row, self.n_col, self.Height, self.Width

    def __len__(self):
        return len(self.samples)
    
    def image_resize(self, image, new_size):
        """
            resize
        """
        h, w = image.shape[:-1]
        if h > w:
            new_h = new_size
            new_w = int(w * new_size / h + 0.5)
        else:
            new_w = new_size
            new_h = int(h * new_size / w + 0.5)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)   # 这里没有写反噢
        return image
    
    def multi_scale_aug(self, image, base_size, rand_scale=1.):
        """
            多尺度的操作
        """
        new_size = int(base_size * rand_scale + 0.5)
        image = self.image_resize(image, new_size)
        return image
    
    def pad_image(self, image, h, w, size, pad_value):
        """
            填充图片
        """
        pad_image = image.copy()  # shadow copy
        pad_h = max(size[0] - h, 0)  # 判断是否需要填充  [h, w, c]
        pad_w = max(size[1] - w, 0)
        if pad_h > 0 or pad_w > 0:  # 右下方填充
            pad_image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=pad_value)  # 边框

        return pad_image
    
    def inference_flip(self, model, image, gpu, flip=False):
        """
            原图先预测一个结果
            flip之后再预测一个结果再flip回来
            二者相加，再除以2得到最终的预测结果
        """
        size = image.size()         # (1, 3, 256, 256)
        pred = model(image)
        pred = F.interpolate(input=pred, size=(size[-2], size[-1]), mode='bilinear', align_corners=True)
        if flip:
            flip_img = image.cpu().numpy()[:, :, :, ::-1]           # 水平翻转
            # flip_output = model(torch.from_numpy(flip_img.copy()).cuda(gpu))###################################################gpu用这个
            flip_output = model(torch.from_numpy(flip_img.copy()))
            flip_output = F.interpolate(input=flip_output, size=(size[-2], size[-1]), mode='bilinear',
                                        align_corners=True)
            flip_pred = flip_output.cpu().numpy().copy()
            # flip_pred = torch.from_numpy(flip_pred[:, :, :, ::-1].copy()).cuda(gpu)######################################gpu用这个
            flip_pred = torch.from_numpy(flip_pred[:, :, :, ::-1].copy())
            pred += flip_pred
            pred = pred * 0.5
        return pred.exp()############################################??????
    
    def multi_scale_inference_resize(self, model, image, gpu, scales=None, flip=False, padding=False):
        """
            多尺度推理
            把输入图像resize到原图的[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]然后再进行推理
            每个尺度推理的结果进行相加
        """
        if scales is None:
            scales = [1]
        batch, _, ori_height, ori_width = image.size()
        assert batch == 1, "only supporting batchsize 1."
        image = image.cpu().numpy()[0].transpose((1, 2, 0)).copy()  # [H, W, C]
        final_pred = torch.zeros([1, self.num_classes, ori_height, ori_width]).cuda(gpu)

        base_size = ori_height
        padvalue = 0.
        if base_size == 2048:
            scales = [0.5, 0.75, 1.0]

        for scale in scales:
            new_img = self.multi_scale_aug(image=image, base_size=base_size, rand_scale=scale)  # resize
            height, width = new_img.shape[:-1]
            if padding:
                new_img = self.pad_image(new_img, height, width, self.crop_size, pad_value=padvalue)
            new_img = new_img.transpose((2, 0, 1))  # [C, H, W]
            new_img = np.expand_dims(new_img, axis=0)  # [B, C, H, W]
            new_img = torch.from_numpy(new_img).cuda(gpu)
            preds = self.inference_flip(model, new_img, gpu, flip)  # direct inference
            preds = preds[:, :, 0:height, 0:width]
            preds = F.interpolate(preds, (ori_height, ori_width), mode='bilinear', align_corners=True)
            final_pred += preds             # 把每个尺度的预测结果都加起来
        return final_pred

def create_filename(input_dir):
    img_filename = []
    names = []
    path_list = os.listdir(input_dir)
    path_list.sort()
    for filename in path_list:
        char_name = filename.split('.')[0]
        names.append(char_name)
        file_path = os.path.join(input_dir, filename)
        img_filename.append(file_path)

    return img_filename, names

def save_pred(preds, Height, Width, name, args):
    print(preds.shape)
    water = preds == 0
    baresoil = preds == 1
    road = preds == 2
    industry = preds == 3
    residential =  preds == 4
    vegetation = preds == 5
    woodland = preds == 6
    plantingarea = preds == 7
    humanbuilt = preds == 8

    h = preds.shape[0]
    w = preds.shape[1]
    del preds
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :, 0] = water * 0 + baresoil * 139 + road * 83 + industry * 255 + residential * 205 + vegetation * 0 + woodland * 0 + plantingarea * 139 + humanbuilt * 189
    rgb[:, :, 1] = water * 0 + baresoil * 0 + road * 134 + industry * 0 + residential * 173 + vegetation * 255 + woodland * 139 + plantingarea * 105 + humanbuilt * 183
    rgb[:, :, 2] = water * 255 + baresoil * 0 + road * 139 + industry * 0 + residential * 0 + vegetation * 0 + woodland * 0 + plantingarea * 20 + humanbuilt * 107
    rgb = rgb[:Height, :Width, :]
    save_img = Image.fromarray(np.uint8(rgb))
    save_img.save(os.path.join(args.save_path, 'pred_' + name + '.png'))

def parse_args():
    """
        配置参数
    """
    parser = argparse.ArgumentParser(description='test segmentation network')
    parser.add_argument('--cfg', help='the path of config file', default='experimentals_yaml/test_w18.yaml')
    # E:/try/land_cover_classification/test_data/image
    parser.add_argument('--data', help='the path of testing data', default='')
    parser.add_argument('--batch_size', help='mini-batch size', default=1)
    parser.add_argument('--classes', help='the number of classes', default=9)
    parser.add_argument('--weights_path', help='the path of test model', default='./model/')
    # hrnet_w18_epoch50_tr_fwiou_0.90565_tr_OA_0.9457_val_fwiou_0.57520_val_OA_0.6494.pth
    parser.add_argument('--weights_name', help='the name of test model weight', default='')
    # E:\try\land_cover_classification\test_data\pre
    parser.add_argument('--save_path', help='the path of save file', default='')
    parser.add_argument('--gpu', help='gpu id to use', default=0, type=int)
    parser.add_argument('--multi_scaled_test', help='if use multi_scaled_test', default=False)
    args = parser.parse_args()
    update_config(config, args)
    return args

def test_model(test_dataset, testloader, model, name, args):
    """
        测试模型
    """
    model.eval()
    n_pred = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(testloader)):
            image, n_row, n_col, Height, Width = batch
            img_shape = image.shape
            # image = image.cuda(args.gpu)
            if args.multi_scaled_test:
                scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
                pred = test_dataset.multi_scale_inference_resize(model, image, args.gpu, scales, flip=True)
            else:
                pred = test_dataset.inference_flip(model, image, args.gpu, flip=True)
            pred = F.interpolate(pred, (img_shape[-2], img_shape[-1]), mode='bilinear', align_corners=True)
            pred = pred.cpu().numpy()
            pred = pred.transpose((0, 2, 3, 1))     # [b, h, w, c]      (1, 256, 256, classes)
            pred = np.argmax(pred, axis=-1)           # (1, 256, 256)
            n_pred.append(pred)
        row = int(n_row.numpy()[0])
        col = int(n_col.numpy()[0])
        stride = 256
        height = (row - 1) * stride + img_shape[-2]
        width = (col - 1) * stride + img_shape[-1]
        pred_np = np.zeros((height, width), dtype=np.uint8)
        print(pred_np.shape)
        for i in range(row):
            row_start = i * stride
            row_end = row_start + img_shape[-2]
            for j in range(col):
                col_start = j * stride
                col_end = col_start + img_shape[-1]
                num = i * col + j  # 第几张图片
                l_0 = num // args.batch_size
                l_1 = num % args.batch_size
                lab = n_pred[l_0][l_1]
                pred_np[row_start:row_end, col_start:col_end] = lab
        del n_pred
        del test_dataset
        del testloader
        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)
        save_pred(pred_np, int(Height.numpy()[0]), int(Width.numpy()[0]), name, args)


def main():
    """
        主函数
    """
    args = parse_args()
    args.data = textEdit_2.toPlainText()
    args.weights_name = textEdit.toPlainText()
    args.save_path = textEdit_3.toPlainText()
    main_worker(args.gpu, args)
    
def main_worker(gpu, args):
    """
        main_worker
    """
    args.gpu = gpu
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True
    begin_time = time.time()

    model = eval(config.MODEL.NAME + '.get_seg_model')(config)

    model_abs_path = os.path.join(args.weights_path, args.weights_name)
    # pretrained_dict = torch.load(model_abs_path)#######################################################################################gpu用这个
    pretrained_dict = torch.load(model_abs_path, map_location='cpu')        # 加载训练好的模型权重
    model_dict = model.state_dict()                 # 原始初始化的模型
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}      # 只取模型定义中有的部分
    model_dict.update(pretrained_dict)                  # 更新整个模型的参数
    model.load_state_dict(model_dict)               # 把更新好的参数加载到模型中
    # model.cuda(args.gpu)#######################################################################################gpu用这个
    print('load model pre-trained done!')

    img_test_path = args.data
    f_names, names = create_filename(img_test_path)
    for i in range(len(names)):
        f_name = f_names[i]
        print(f_name)
        test_dataset = TestDataLoader(f_name)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=0)
        name = names[i]
        test_model(test_dataset, test_loader, model, name, args)

    end_time = time.time()
    time_used = end_time - begin_time
    print('Inference time : {}s'.format(time_used))

if __name__ == '__main__':
    app = QApplication([])
    window = QMainWindow()
    window.resize(300, 300)
    window.move(300, 310)
    window.setWindowTitle('GF3-SAR_testing')

    textEdit_2 = QPlainTextEdit(window)
    textEdit_2.setPlaceholderText("testing data path")
    textEdit_2.move(10,25)
    textEdit_2.resize(250,50)

    textEdit = QPlainTextEdit(window)
    textEdit.setPlaceholderText("weights_name")
    textEdit.move(10,105)
    textEdit.resize(250,50)

    textEdit_3 = QPlainTextEdit(window)
    textEdit_3.setPlaceholderText("save_path")
    textEdit_3.move(10,205)
    textEdit_3.resize(250,50)

    button = QPushButton('testing', window)
    button.move(150,180)
    button.clicked.connect(main)

    window.show()
    app.exec_()
    # main()
