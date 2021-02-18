
from model_utils import *


def unet(pretrained_weights = None,input_size = (384,384,1)):
    image = Input(input_size)
    kernel_size= 5
    ch= 32
    act= 'relu'
    pad= 'same'
    init= 'he_normal'
    conv1= Conv2D(ch, kernel_size, activation=act, padding=pad, kernel_initializer=init)(image)
    conv1= Conv2D(ch, kernel_size, activation=act, padding=pad, kernel_initializer=init)(conv1)
    pool1= MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1= Dropout(0.5)(pool1)

    conv2= Conv2D(ch*2, kernel_size, activation=act, padding=pad, kernel_initializer=init)(pool1)
    conv2= Conv2D(ch*2, kernel_size, activation=act, padding=pad, kernel_initializer=init)(conv2)
    pool2= MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2= Dropout(0.5)(pool2)

    conv3= Conv2D(ch*4, kernel_size, activation=act, padding=pad, kernel_initializer=init)(pool2)
    conv3= Conv2D(ch*4, kernel_size, activation=act, padding=pad, kernel_initializer=init)(conv3)
    pool3= MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3= Dropout(0.5)(pool3)

    conv4= Conv2D(ch*8, kernel_size, activation=act, padding=pad, kernel_initializer=init)(pool3)
    conv4= Conv2D(ch*8, kernel_size, activation=act, padding=pad, kernel_initializer=init)(conv4)
    pool4= MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4= Dropout(0.5)(pool4)

    conv5= Conv2D(ch*16, kernel_size, activation=act, padding=pad, kernel_initializer=init)(pool4)
    conv5= Conv2D(ch*16, kernel_size, activation=act, padding=pad, kernel_initializer=init)(conv5)

    up6= UpSampling2D(size = (2,2))(conv5)
    up6= Conv2D(ch*8, 2, activation=act, padding=pad, kernel_initializer=init)(up6)
    merge6= concatenate([conv4,up6], axis = 3)
    merge6= Dropout(0.5)(merge6)
    conv6= Conv2D(ch*8, kernel_size, activation=act, padding=pad, kernel_initializer=init)(merge6)
    conv6= Conv2D(ch*8, kernel_size, activation=act, padding=pad, kernel_initializer=init)(conv6)

    up7= UpSampling2D(size = (2,2))(conv6)
    up7= Conv2D(ch*4, 2, activation=act, padding=pad, kernel_initializer=init)(up7)
    merge7= concatenate([conv3,up7], axis = 3)
    merge7= Dropout(0.5)(merge7)
    conv7= Conv2D(ch*4, kernel_size, activation=act, padding=pad, kernel_initializer=init)(merge7)
    conv7= Conv2D(ch*4, kernel_size, activation=act, padding=pad, kernel_initializer=init)(conv7)

    up8= Conv2D(ch*2, 2, activation=act, padding=pad, kernel_initializer=init)(UpSampling2D(size = (2,2))(conv7))
    merge8= concatenate([conv2,up8], axis = 3)
    merge8= Dropout(0.5)(merge8)
    conv8= Conv2D(ch*2, kernel_size, activation=act, padding=pad, kernel_initializer=init)(merge8)
    conv8= Conv2D(ch*2, kernel_size, activation=act, padding=pad, kernel_initializer=init)(conv8)

    up9= Conv2D(ch, 2, activation=act, padding=pad, kernel_initializer=init)(UpSampling2D(size = (2,2))(conv8))
    merge9= concatenate([conv1,up9], axis = 3)
    merge9= Dropout(0.5)(merge9)
    conv9= Conv2D(ch, kernel_size, activation=act, padding=pad, kernel_initializer=init)(merge9)
    conv9= Conv2D(ch, kernel_size, activation=act, padding=pad, kernel_initializer=init)(conv9)
    conv9= Conv2D(5, 1, activation='softmax', padding=pad, kernel_initializer=init)(conv9)

    model= Model(inputs = image, outputs = conv9)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'categorical_crossentropy', metrics = ['accuracy', dice, dice_fb, dice_fc, dice_tb, dice_tc])

    return model
