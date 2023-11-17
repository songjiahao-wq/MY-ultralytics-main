import imutils
import os
import cv2
img_path = r'D:\songjiahao\DATA\smokke\VOC\AUG\Augimages2/'
label_path = r'D:\songjiahao\DATA\smokke\VOC\AUG\labels/'
f = os.listdir(img_path)
label_id = []
def paint(label_file, img_file):
    #读取照片
    img = cv2.imread(img_file)

    img_h, img_w, _ = img.shape
    with open(label_file, 'r') as f:
        obj_lines = [l.strip() for l in f.readlines()]
    for obj_line in obj_lines:
        cls, cx, cy, nw, nh = [float(item) for item in obj_line.split(' ')]
        color = (0, 0, 255) if cls == 0.0 else (0, 255, 0)
        x_min = int((cx - (nw / 2.0)) * img_w)
        y_min = int((cy - (nh / 2.0)) * img_h)
        x_max = int((cx + (nw / 2.0)) * img_w)
        y_max = int((cy + (nh / 2.0)) * img_h)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(img,obj_line[0:2],(x_min,y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        # if int(obj_line[0:2]) not in label_id:
        #     label_id.append(int(obj_line[0:2]))

    # img2 = imutils.resize(img,width=1600)
    cv2.imshow('Ima', img)
    # print(sorted(label_id))
    cv2.waitKey(0)

    cv2.destroyAllWindows()
for i in f:
    if i.split('.')[-1] =='jpg':
        label_path_name = label_path + i.replace('.jpg','.txt')
    elif i.split('.')[-1] == 'png':
        label_path_name = label_path + i.replace('.png', '.txt')

    img_path_name = img_path + i
    # print(img_path_name,label_path_name)
    # print(img_path_name)
    if label_path_name.endswith('.txt') and img_path_name.endswith('.jpg'):
        paint(label_path_name,img_path_name)
