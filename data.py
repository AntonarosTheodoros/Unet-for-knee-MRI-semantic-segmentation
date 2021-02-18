from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import glob
import skimage.io as io
import skimage.transform as trans
import tensorflow as tf

count=0

def dataGenerator(batch_size,train_path,image_folder,mask_folder, aug = dict(rescale= 1./255),
                    image_color_mode = "grayscale", mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (384,384),seed = 1, save=0, steps=0):
    image_datagen = ImageDataGenerator(**aug)
    mask_datagen = ImageDataGenerator(**aug)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = None,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    global count
    data_generator = zip(image_generator, mask_generator)
    for (img,mask) in data_generator:
        new_mask= multi_channel(mask)
        if save==1 and count<steps:
            savedata('test/image', 'test/mask', img, new_mask)
        yield (img,new_mask)


def multi_channel(image):
    y= np.round(image*4)
    fb= np.equal(y,1).astype(int)
    fc= np.equal(y,2).astype(int)
    tb= np.equal(y,3).astype(int)
    tc= np.equal(y,4).astype(int)
    z= 1 - (fb + fc + tb + tc)
    new= np.concatenate((z, fb, fc, tb, tc), axis=3)
    return new

def savedata(img_path, mask_path, img_npyfile, mask_npyfile):
    global count
    for i,item in enumerate(img_npyfile):
        img= img_npyfile[i,:,:,:]
        img= (img*255).astype(np.uint8)
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        io.imsave(os.path.join(img_path, "%d.png"%(count*8 +i)),img)
    for i,item in enumerate(mask_npyfile):
        mask= mask_npyfile[i,:,:,:]
        img= np.argmax(mask, axis=2)/4
        img= (img*255).astype(np.uint8)
        if not os.path.exists(mask_path):
            os.makedirs(mask_path)
        io.imsave(os.path.join(mask_path, "%d.png"%(count*8 +i)),img)
    count= count +1

def savePredict(save_path,npyfile):
    for i,item in enumerate(npyfile):
        y= npyfile[i,:,:,:]
        img= np.argmax(y ,axis=2)/4
        img= (img*255).astype(np.uint8)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        io.imsave(os.path.join(save_path,"%d.png"%i),img)
