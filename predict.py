from model import *
from data import *
from model_utils import *
import keras.metrics
from dicePost import *
from colorFill import *

keras.metrics.dice = dice
keras.metrics.dice_b = dice_fb
keras.metrics.dice_fc = dice_fc
keras.metrics.dice_tc = dice_tb
keras.metrics.dice_tc = dice_tc
keras.losses.weighted_crossentropy= weighted_crossentropy

model= load_model('test/segmentation.model')

#Prediction
predictGene = dataGenerator(8, 'MyDataset/70-15-15/Regression/test', 'image', 'mask', save=1, steps=5, seed= 2)
results = model.predict_generator(predictGene,5, verbose=1)
savePredict('test/predict',results)

#PostProcess
