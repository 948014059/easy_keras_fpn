from PIL import Image
import os
import  numpy as np
from keras.preprocessing import image
import cv2
def binarylab(labels,classes,size):
    y=np.zeros((size,size,classes))
    for i in range(size):
        for j in range(size):
            y[i,j,int(labels[i][j])]=1
    return y


def load_data(split=0.9):
    img_dir = 'dataset/img'
    label_dir = 'dataset/label'
    label_paths = [os.path.join(label_dir, path) for path in os.listdir(label_dir)]
    image_paths = [os.path.join(img_dir, path) for path in os.listdir(img_dir)]

    train_len=int(len(label_paths)*split)

    x_train=image_paths[:train_len]
    y_train=label_paths[:train_len]
    x_test=image_paths[train_len:]
    y_test=label_paths[train_len:]
    print('train on %s, test on %s '%(train_len,(len(image_paths)-train_len)))

    return x_train,y_train,x_test,y_test

def gendata(x_data,y_data,batch_size):
    while True:
        data=[]
        label=[]
        for index,img_path in enumerate(x_data):
            img=cv2.imread(img_path)
            img=cv2.resize(img,(416,416))
            data.append(img)

            label_img=Image.open(y_data[index])
            label_img = label_img.resize([416, 416])
            label_img = image.img_to_array(label_img)
            label_ = np.reshape(label_img, [416, 416])
            mask = label_ == 255
            label_[mask] = 0
            y = binarylab(label_, 3, 416)
            y = np.expand_dims(y, axis=0)
            label.append(y)
            if len(data)==batch_size:
                data=np.array(data).reshape(-1,416,416,3)
                label=np.array(label).reshape(-1,416,416,3)
                yield  data,label
                data=[]
                label=[]

    # print(y.shape)
    # with open('lebel.txt','w',encoding='utf-8') as f:
    #     f.write(str(np.array(y).tolist()))
    # img.show()
    # img_.show()
    # print(np.array(img))
# x_train,y_train,x_test,y_test=load_data()
# gen=gendata(x_train,y_train,1)
# data,label=next(gen)
# print(data.shape,label.shape)
