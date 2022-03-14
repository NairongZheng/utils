import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

# 获取全部类别标签
classes = []
def gen_classes(image_id):
    in_file = open('%s/%s.xml'%(path,image_id))
    tree = ET.parse(in_file)
    root = tree.getroot()    
    for obj in root.iter('object'):
        cls_name = obj.find('name').text
        if cls_name in classes:
            pass
        else:
            classes.append(cls_name)
    return classes

def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 
    y = (box[2] + box[3])/2.0 
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = round(x*dw, 6)
    w = round(w*dw, 6)
    y = round(y*dh, 6)
    h = round(h*dh, 6)
    return (x,y,w,h)

def convert_annotation(in_path,out_path,image_name):
    in_file = open('%s/%s.xml'%(in_path,image_name))
    out_file = open('%s/%s.txt'%(out_path,image_name), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

# path = 'C:/Users/zy080/Downloads/水面漂浮物数据集-2400/VOCdevkit/VOC2007'
# sets = ['train','test','val']
# path = 'D:/data/AI_car/quchongout/JwaQidpW8OSLXqdgIId/label_data'
# for image_set in sets:
#     if not os.path.exists('%s/%s'%(path,image_set)+'labels/'):
#         os.makedirs('%s/%s'%(path,image_set)+'labels/')
#     image_ids = open('%s/ImageSets/Main/%s.txt'%(path,image_set)).read().strip().split()
#     for image_id in image_ids:
#         gen_classes(image_id)
#         convert_annotation(image_set,image_id)
#     classes_file = open('%s/%s'%(path,image_set)+'labels/classes.txt','w')
#     classes_file.write("\n".join([a for a in classes]))
#     classes_file.close()

if __name__ == '__main__':
    path = r'E:\data\detection\SSDD\Annotations'
    out_path = r'E:\data\detection\SSDD\labeltxt'
    label_list = os.listdir(path)
    label_names = [i.split('.')[0] for i in label_list]
    for id in label_names:
        gen_classes(id)
        convert_annotation(path, out_path, id)
    classes_file = open(path + 'classes.txt', 'w')
    classes_file.write("\n".join([a for a in classes]))
    classes_file.close()

    print('ok')
