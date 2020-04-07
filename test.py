import  os
import cv2
import tensorflow as tf
import numpy as np
import random
img_dir = r'E:\DataSets\kzsb\test'
label_dir = 'dataset/label'
label_paths = [os.path.join(label_dir, path) for path in os.listdir(label_dir)]
image_paths = [os.path.join(img_dir, path) for path in os.listdir(img_dir)]
model=tf.keras.models.load_model('fcnface1.h5')

random.shuffle(image_paths)
corlor=np.array([(0,0,0),(0,255,0),(0,0,255)])


for image_path in image_paths:
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img_=img.copy()
    img = img.reshape(-1,224,224,3)
    pred=model.predict(img)
    prediction = np.argmax(pred, axis=-1)
    with open('pred.txt','w',encoding='utf8') as f :
        f.write(str(prediction.tolist()))

    color_image = np.array(corlor)[prediction.ravel()].reshape(224, 224, 3).astype('uint8')
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    image=cv2.erode(color_image,kernel)

    dst = cv2.addWeighted(img_, 1, image, 0.8, 0)
    dst=cv2.resize(dst,(416,416))
    cv2.imshow('s', dst)
    cv2.waitKey(0)
