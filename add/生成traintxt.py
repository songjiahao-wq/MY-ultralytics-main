import os
import random
import sys
"""
    labels-xml:
        --trian
        --test
        --val
        --ImageSets
            --train.txt
            --test.txt
            --val.txt
"""
# root_path = sys.argv[1]
convert_xml = ['train','test','val']
for convert_name in convert_xml:
    root_path = r'D:/BaiduNetdiskDownload/NEU-DET/xml'
    xmlfilepath = root_path + '/'+convert_name#更改
    txtsavepath = root_path + '/ImageSets'
    if not os.path.exists(txtsavepath):
        os.makedirs(txtsavepath)
    if not os.path.exists(xmlfilepath):
        os.makedirs(xmlfilepath)
    total_xml = os.listdir(xmlfilepath)
    num = len(total_xml)
    list = range(num)
    ftrainval = open(txtsavepath + '/'+convert_name+'.txt', 'w')
    for i in list:
        name = xmlfilepath +'/'+ total_xml[i] + '\n'
        print(name)
        ftrainval.write(name)
    ftrainval.close()