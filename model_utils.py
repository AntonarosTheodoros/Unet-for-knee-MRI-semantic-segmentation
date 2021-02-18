import tensorflow as tf
import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras import backend as K
import keras


def weighted_crossentropy(y_true, y_pred):
    z_true, b_true, fc_true, tc_true= tf.split(y_true, 4, axis=3)
    sum= K.sum(K.flatten(y_true))
    n= K.sum(K.flatten(b_true))
    w1= (n+1)/(K.sum(K.flatten(fc_true)) +1)
    w2= (n+1)/(K.sum(K.flatten(tc_true)) +1)
    fc_true= w1*fc_true
    tc_true= w2*tc_true
    y_true = tf.concat((z_true, b_true, fc_true, tc_true), axis=3)
    y_pred= K.clip(y_pred, K.epsilon(), 1)
    wce= -y_true*K.log(y_pred)
    loss= K.sum(K.flatten(wce))/sum
    return loss


def dice(y_true, y_pred):
    y_true_f= K.flatten(y_true)
    y_pred_f= K.flatten(y_pred)
    numerator= 2*K.sum(y_true_f * y_pred_f)
    denominator= K.sum(y_true_f) + K.sum(y_pred_f)
    dice_metric= (numerator + 1) / (denominator + 1)
    return  dice_metric


def dice_fb(y_true, y_pred):
    z_true, fb_true, fc_true, tb_true, tc_true= tf.split(y_true, 5, axis=3)
    z_pred, fb_pred, fc_pred, tb_pred, tc_pred= tf.split(y_pred, 5, axis=3)
    dicefb= dice(fb_true, fb_pred)
    return  dicefb

def dice_fc(y_true, y_pred):
    z_true, fb_true, fc_true, tb_true, tc_true= tf.split(y_true, 5, axis=3)
    z_pred, fb_pred, fc_pred, tb_pred, tc_pred= tf.split(y_pred, 5, axis=3)
    dicefc= dice(fc_true, fc_pred)
    return  dicefc

def dice_tb(y_true, y_pred):
    z_true, fb_true, fc_true, tb_true, tc_true= tf.split(y_true, 5, axis=3)
    z_pred, fb_pred, fc_pred, tb_pred, tc_pred= tf.split(y_pred, 5, axis=3)
    dicetb= dice(tb_true, tb_pred)
    return  dicetb

def dice_tc(y_true, y_pred):
    z_true, fb_true, fc_true, tb_true, tc_true= tf.split(y_true, 5, axis=3)
    z_pred, fb_pred, fc_pred, tb_pred, tc_pred= tf.split(y_pred, 5, axis=3)
    dicetc= dice(tc_true, tc_pred)
    return  dicetc

def lrs_function(epoch, lr):
    if (epoch+1)/20 == (epoch+1)//20:
        lr= lr/2
    return lr
