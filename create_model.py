from tensorflow import keras
from tensorflow.keras.layers import *
from load_data import load_data,gendata

def create_model():
    input=keras.Input((224,224,3))
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input)
    conv1 =BatchNormalization()(conv1)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPool2D((2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2=BatchNormalization()(conv2)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPool2D((2, 2))(conv2)

    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3=BatchNormalization()(conv3)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)

    up1 = concatenate([UpSampling2D((2, 2))(conv3), conv2], axis=-1)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(up1)
    conv4=BatchNormalization()(conv4)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)

    up2 = concatenate([UpSampling2D((2, 2))(conv4), conv1], axis=-1)


    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(up2)
    conv5=BatchNormalization()(conv5)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5)

    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)

    # h = Reshape([224 * 224, 3])(conv5)
    # h=Permute([2,1])(h)
    # h=Activation('softmax')(h)
    # out=Reshape([224,224,3])(h)
    out = Conv2D(3, (1, 1), padding='same',activation='softmax')(conv5)
    model=keras.models.Model(inputs=input,outputs=out)
    return model

batch_size=2
model=create_model()
# model=keras.models.load_model('fcnface1.h5')
keras.utils.plot_model(model,show_shapes=True,to_file='fcn.png')
x_train,y_train,x_test,y_test=load_data(split=0.8)
# print(x_train.shape,y_train.shape)
model.compile(optimizer=keras.optimizers.Adam(lr=3e-6),loss='categorical_crossentropy',metrics=['acc'])
model.fit_generator(gendata(x_train,y_train,batch_size),
                    steps_per_epoch=len(x_train)//batch_size,
                    validation_data=gendata(x_test,y_test,batch_size),
                    validation_steps=len(x_test)//batch_size,
                    epochs=50)
model.save('fcnface1.h5')

