#!/usr/bin/env python
# -*- coding:utf-8 -*-
import cv2
def showPrioriBox():
	#输入图片尺寸
	INPUT_SIZE = 416

	mask52=	[0,1,2]
	mask26=	[3,4,5]
	mask13=	[6,7,8]
	anchors=[ 10,13, 16,30, 33,23,
			  30,61, 62,45, 59,119,
			  116,90, 156,198, 373,326]
	FEATURE_MAP_SIZE=26
	SHOW_ALL_FLAG  = True  # 显示所有的方框
	GRID_SHOW_FLAG =True

	# cap = cv2.VideoCapture("street.jpg")
	picPath = r'D:\yanyi\xianyu\yolov5-MY\data\images\bus.jpg'
	picName = picPath.split('/')[-1]
	img = cv2.imread(picPath)
	print("original img.shape: ",img.shape)  # (1330, 1330, 3)
	img = cv2.resize(img,(INPUT_SIZE, INPUT_SIZE))

	# 显示网格
	if GRID_SHOW_FLAG:
		height, width, channels = img.shape
		GRID_SIZEX =  int(INPUT_SIZE/FEATURE_MAP_SIZE)
		for x in range(0, width - 1, GRID_SIZEX):
			cv2.line(img, (x, 0), (x, height), (150, 150, 255), 1, 1)  # x grid

		GRID_SIZEY = int(INPUT_SIZE / FEATURE_MAP_SIZE)
		for y in range(0, height - 1, GRID_SIZEY):
			cv2.line(img, (0, y), (width, y), (150, 150, 255), 1, 1)  # x grid
		# END：显示网格
		# cv2.imshow('Hehe', img)
		# cv2.imwrite('./' + picName.split('.')[0] + '_grid.' + picName.split('.')[1], img)

	if SHOW_ALL_FLAG or FEATURE_MAP_SIZE==13:
		for ele in mask13:
			# print(ele)
			cv2.rectangle(img, (
			(int(INPUT_SIZE * 0.5 - 0.5*anchors[ ele * 2]), int(INPUT_SIZE * 0.5 -  0.5*anchors[ ele * 2 + 1]))),
						  ((int(INPUT_SIZE * 0.5 + 0.5*anchors[ ele * 2]),
							int(INPUT_SIZE * 0.5 + 0.5*anchors[ ele * 2 + 1]))), (0, 255-ele*10, 0), 2)
		# cv2.imwrite('./' + picName.split('.')[0] + '_saveMask13.' + picName.split('.')[1], img)

	if SHOW_ALL_FLAG or FEATURE_MAP_SIZE==26:
		for ele in mask26:
			# print(ele)
			cv2.rectangle(img, (
			(int(INPUT_SIZE * 0.5 - 0.5*anchors[ ele * 2]), int(INPUT_SIZE * 0.5 -  0.5*anchors[ ele * 2 + 1]))),
						  ((int(INPUT_SIZE * 0.5 + 0.5*anchors[ ele * 2]),
							int(INPUT_SIZE * 0.5 + 0.5*anchors[ ele * 2 + 1]))), (255, 255-ele*10, 0), 2)
		# cv2.imwrite('./' + picName.split('.')[0] + '_saveMask26.' + picName.split('.')[1], img)

	if SHOW_ALL_FLAG or FEATURE_MAP_SIZE==52:
		for ele in mask52:
			# print(ele)
			cv2.rectangle(img, (
			(int(INPUT_SIZE * 0.5 - 0.5*anchors[ ele * 2]), int(INPUT_SIZE * 0.5 -  0.5*anchors[ ele * 2 + 1]))),
						  ((int(INPUT_SIZE * 0.5 + 0.5*anchors[ ele * 2]),
							int(INPUT_SIZE * 0.5 + 0.5*anchors[ ele * 2 + 1]))), (0, 255-ele*10, 255), 1)
		# cv2.imwrite('./' + picName.split('.')[0] + '_saveMask52.' + picName.split('.')[1], img)


	cv2.imwrite('./' + picName.split('.')[0] + '_allSave.' + picName.split('.')[1], img)

	cv2.imshow('img', img)
	while cv2.waitKey(1000) != 27:  # loop if not get ESC.
		if cv2.getWindowProperty('img', cv2.WND_PROP_VISIBLE) <= 0:
			break
	cv2.destroyAllWindows()

if __name__ == '__main__':
    showPrioriBox()
