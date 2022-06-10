"""
    把2dhrnet模型复制成3d
"""
import argparse
import os
import numpy as np

import torch

import torch.backends.cudnn as cudnn

from hrnet_3d import get_seg_model
from default import _C as config
from default import update_config


def parse_args():
    """
        配置参数
    """
    parser = argparse.ArgumentParser(description='test segmentation network')
    parser.add_argument('--cfg', help='the path of config file', default='/emwuser/znr/code/hyper_sar/experimentals_yaml/test_w18.yaml')
    # parser.add_argument('--data', help='the path of testing data', default='/emwuser/znr/data/hyper_sar/test/img')
    parser.add_argument('--batch_size', help='mini-batch size', default=1)
    parser.add_argument('--classes', help='the number of classes', default=9)
    parser.add_argument('--weights_path', help='the path of test model', default='/emwuser/znr/code/yyjf/pretrained')
    parser.add_argument('--weights_name', help='the name of test model weight', default='hrnetv2_w18_imagenet_pretrained.pth')
    # parser.add_argument('--save_path', help='the path of save file', default='/emwuser/znr/data/hyper_sar/test/pre/20220606_up_xiongan')
    parser.add_argument('--gpu', help='gpu id to use', default=0, type=int)
    parser.add_argument('--multi_scaled_test', help='if use multi_scaled_test', default=False)
    args = parser.parse_args()
    update_config(config, args)
    return args


def main():
    """
        主函数
    """
    args = parse_args()
    main_worker(args.gpu, args)

def main_worker(gpu, args):
    """
        main_worker
    """
    args.gpu = gpu
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True

    model = get_seg_model(config)

    # model = ResUNet.ResUNet(out_channel=config.DATASET.NUM_CLASSES)

    model_abs_path = os.path.join(args.weights_path, args.weights_name)
    pretrained_dict = torch.load(model_abs_path)        # 加载训练好的模型权重
    model_dict = model.state_dict()                 # 原始初始化的模型
    new_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys():
            if pretrained_dict[k].shape != model_dict[k].shape:
                new_val = v.numpy()
                new_val = np.expand_dims(new_val, axis=2).repeat(model_dict[k].shape[2], axis=2)
                new_dict[k] = torch.Tensor(new_val)
    torch.save(new_dict, '/emwuser/znr/code/yyjf/pretrained/hrnetv2_3d_w18_imagenet_pretrained.pth')
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}      # 只取模型定义中有的部分
    # model_dict.update(pretrained_dict)                  # 更新整个模型的参数
    # model.load_state_dict(model_dict)               # 把更新好的参数加载到模型中
    # model.cuda(args.gpu)

if __name__ == '__main__':
    main()
