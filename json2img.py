import json
import os
import numpy as np
import cv2
from tqdm import tqdm
img_dir='dataset/img'
label_dir='dataset/label'

json_dir='E:\FTP\datasets\json_'
json_paths=[os.path.join(json_dir,path) for path in os.listdir(json_dir)]

# 将labelme标注后的json文件转换为label
label_names=['face','face_mask']
for json_path in  tqdm(json_paths):
    name=json_path.split('.')[0].split('\\')[-1]
    # print(name)
    img_path=json_path.replace('json_','JPEGImages').replace('json','jpg')
    image=cv2.imread(img_path)
    # print(image)
    with open(json_path,'r',encoding='utf8') as f:
        json_data=json.load(f)
    label_len=json_data['shapes']
    wigth = json_data['imageWidth']
    height = json_data['imageHeight']
    # img = np.zeros((height, wigth, 3), np.uint8)
    img = np.zeros((height, wigth, 1), np.uint8)
    for label_ in label_len:
        # print(label_)
        lb=label_['label']
        points=label_['points']
        pts = np.array(points, np.int32)
        for index ,value in enumerate(label_names):
            if lb == value:
                # print('aa')
                # print(value,index+1)
                cv2.fillPoly(img,[pts],(index+1))
                # cv2.fillPoly(img, [pts], (0,255,255))
    cv2.imwrite(os.path.join(img_dir,'%s.jpg'%name),image)
    cv2.imwrite(os.path.join(label_dir,'%s.png'%name),img)


    # image=cv2.resize(image,(224,224))
    # img=cv2.resize(img,(224,224))
    # dst = cv2.addWeighted(image, 1, img.astype('uint8'), 0.5, 0)
    # with open('test.txt','w',encoding='utf8') as f:
    #     f.write(str(img.tolist()))
    #
    #
    # cv2.imshow('img',img*100.)
    # cv2.waitKey(0)
