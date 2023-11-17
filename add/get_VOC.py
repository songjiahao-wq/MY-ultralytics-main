# VOC数据集提取某个类或者某些类
# !/usr/bin/env python
# -*- encoding: utf-8 -*-
import os
import xml.etree.ElementTree as ET
import shutil

# 根据自己的情况修改相应的路径
ann_filepath = 'D:/dataset_copy2/ALL_VOC\dataset/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/Annotations/'
img_filepath = 'D:/dataset_copy2/ALL_VOC\dataset/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/'
img_savepath = 'D:/dataset_copy2/ALL_VOC\dataset/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages_ssd/'
ann_savepath = 'D:/dataset_copy2/ALL_VOC\dataset/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/Annotations_ssd/'
if not os.path.exists(img_savepath):
    os.mkdir(img_savepath)

if not os.path.exists(ann_savepath):
    os.mkdir(ann_savepath)

# 这是VOC数据集中所有类别
# classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
#             'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
#              'dog', 'horse', 'motorbike', 'pottedplant',
#           'sheep', 'sofa', 'train', 'person','tvmonitor']

classes = ['car', 'bus', 'bicycle', 'person', 'motorbike']  # 这里是需要提取的类别

def save_annotation(file):
    tree = ET.parse(ann_filepath + '/' + file)
    root = tree.getroot()
    result = root.findall("object")
    bool_num = 0
    for obj in result:
        if obj.find("name").text not in classes:
            root.remove(obj)
        else:
            bool_num = 1
    if bool_num:
        tree.write(ann_savepath + file)
        return True
    else:
        return False

def save_images(file):
    name_img = img_filepath + os.path.splitext(file)[0] + ".jpg"
    shutil.copy(name_img, img_savepath)
    # 文本文件名自己定义，主要用于生成相应的训练或测试的txt文件
    with open('train.txt', 'a') as file_txt:
        file_txt.write(os.path.splitext(file)[0])
        file_txt.write("\n")
    return True


if __name__ == '__main__':
    for f in os.listdir(ann_filepath):
        print(f)
        if save_annotation(f):
            save_images(f)