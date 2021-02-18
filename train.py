from model import *
from data import *
import time
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


trainGene= dataGenerator(8,'MyDataset/70-15-15/Regression/train','image','mask')
testGene = dataGenerator(8,'MyDataset/70-15-15/Regression/test', 'image', 'mask',seed=2)
valGene  = dataGenerator(8,'MyDataset/70-15-15/Regression/validation','image','mask')
model= unet()

#callbacks
name= "Unet-{}".format(int(time.time()))
tensorboard_callback= TensorBoard(log_dir='Logs/logs{}'.format(name),histogram_freq=1, write_images=True)

#Training and Validation
model.fit_generator(trainGene, steps_per_epoch=550, epochs=50, callbacks=[tensorboard_callback], validation_data= valGene, validation_steps= 100, validation_freq= 1)

#Testing
test_loss, test_accuracy, test_dice, test_dice_fb, test_dice_fc, test_dice_tb, test_dice_tc = model.evaluate_generator(testGene, steps= 100, verbose=1)
print('test_loss: ', np.float16(test_loss), ' - test_accuracy: ', np.float16(test_accuracy), ' - test_dice: ', np.float16(test_dice),
 ' - test_dice_fb: ', np.float16(test_dice_fb), ' - test_dice_fc: ', np.float16(test_dice_fc), ' - test_dice_tb: ', np.float16(test_dice_tb), ' - test_dice_tc: ', np.float16(test_dice_tc))

#Prediction
predictGene = dataGenerator(8, 'MyDataset/70-15-15/Regression/test', 'image', 'mask', save=1, steps=5, seed=2)
results = model.predict_generator(predictGene,5, verbose=1)
savePredict('test/predict',results)
model.save('test/segmentation.model')

#PostProcess
