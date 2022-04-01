"""
    Author:DamonZheng
    Function:json2xml
    Edition:1.0
    Date:2022.4.1
"""

import argparse
import glob
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
import json
from tqdm import tqdm

def parse_args():
    """
        参数配置
    """
    parser = argparse.ArgumentParser(description='json2xml')
    parser.add_argument('--raw_label_dir', help='the path of raw label', default=r'D:\code_python\play_play\data\label_json')
    parser.add_argument('--pic_dir', help='the path of picture', default=r'D:\code_python\play_play\data\img')
    parser.add_argument('--pic_ext', help='the extention of picture', default='.jpg')
    parser.add_argument('--pic_channel', help='the channel number of picture', default='3')
    parser.add_argument('--save_dir', help='the path of new label', default=r'D:\code_python\play_play\data\label')
    args = parser.parse_args()
    return args

def read_json_info(json_path):
    with open(json_path, 'r', encoding='utf8') as f:
        json_data = json.load(f)
        width = json_data['imageWidth']
        height = json_data['imageHeight']
        all_obj = []
        for i in json_data['shapes']:
            obj = {}
            obj['name'] = i['label']
            obj['xmin'] = str(i['points'][0][0])
            obj['ymin'] = str(i['points'][0][1])
            obj['xmax'] = str(i['points'][3][0])
            obj['ymax'] = str(i['points'][3][1])
            # # 下面这三个是为了生成跟参考文件格式一样的, 其实不需要
            # obj['pose'] = 'Unspecified'
            # obj['truncated'] = '0'
            # obj['difficult'] = '0'
            all_obj.append(obj)
        return str(width), str(height), all_obj

def create_node(xml_data, node_name, node_text):
    node = xml_data.createElement(node_name)
    node_t = xml_data.createTextNode(node_text)
    node.appendChild(node_t)
    return node

def main():
    """
        主函数
    """
    args = parse_args()
    labels = glob.glob(args.raw_label_dir + '/*.json')
    for i, label_abs in tqdm(enumerate(labels), total=len(labels)):
        _, label = os.path.split(label_abs)
        label_name = label.rstrip('.json')
        img_path = os.path.join(args.pic_dir, label_name + args.pic_ext)
        img_folder = os.path.basename(args.pic_dir)

        # 读取json中的信息(根据生成的xml需要的信息读取的)
        width, height, all_obj = read_json_info(label_abs)

        # 创建xml信息, 生成xml文件
        xml_data = minidom.Document()               # 创建xml文件

        root = xml_data.createElement('annotation') # 创建annotation节点
        xml_data.appendChild(root)                  # 将annotation节点挂到xml文件中

        # folder = xml_data.createElement('folder')   # 创建folder节点
        # folder_text = xml_data.createTextNode(img_folder)
        # folder.appendChild(folder_text)
        # root.appendChild(folder)
        # filename = xml_data.createElement('filename')   # 创建filename节点
        # filename_text = xml_data.createTextNode(label_name)
        # filename.appendChild(filename_text)
        # root.appendChild(filename)
        folder = create_node(xml_data, 'folder', img_folder)        # 添加folder节点
        root.appendChild(folder)

        filename = create_node(xml_data, 'filename', label_name)    # 添加filename节点
        root.appendChild(filename)

        path = create_node(xml_data, 'path', img_path)              # 添加path节点
        root.appendChild(path)

        source = xml_data.createElement('source')                   # 添加source
        root.appendChild(source)
        database = create_node(xml_data, 'database', 'Unknown')
        source.appendChild(database)

        size = xml_data.createElement('size')                       # 添加size
        root.appendChild(size)
        Width = create_node(xml_data, 'width', width)
        Height = create_node(xml_data, 'height', height)
        depth = create_node(xml_data, 'depth', args.pic_channel)
        size.appendChild(Width)
        size.appendChild(Height)
        size.appendChild(depth)

        segmented = create_node(xml_data, 'segmented', '0')         # 添加segmented
        root.appendChild(segmented)

        for obj in all_obj:
            objects = xml_data.createElement('object')                  # 添加object
            root.appendChild(objects)
            name = create_node(xml_data, 'name', obj['name'])
            pose = create_node(xml_data, 'pose', 'Unspecified')
            truncated = create_node(xml_data, 'truncated', '0')
            difficult = create_node(xml_data, 'difficult', '0')
            objects.appendChild(name)
            objects.appendChild(pose)
            objects.appendChild(truncated)
            objects.appendChild(difficult)
            bndbox = xml_data.createElement('bndbox')
            objects.appendChild(bndbox)
            xmin = create_node(xml_data, 'xmin', obj['xmin'])
            ymin = create_node(xml_data, 'ymin', obj['ymin'])
            xmax = create_node(xml_data, 'xmax', obj['xmax'])
            ymax = create_node(xml_data, 'ymax', obj['ymax'])
            bndbox.appendChild(xmin)
            bndbox.appendChild(ymin)
            bndbox.appendChild(xmax)
            bndbox.appendChild(ymax)
        save_name = os.path.join(args.save_dir, label_name + '.xml')
        with open(save_name, 'wb') as f:
            f.write(xml_data.toprettyxml(encoding='utf-8', indent='    '))

if __name__ == '__main__':
    main()
