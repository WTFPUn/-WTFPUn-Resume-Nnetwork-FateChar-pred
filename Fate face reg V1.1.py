from PIL import  Image
from matplotlib import image
from matplotlib import pyplot
import os
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

y_saber = 1
x_saber = f'Saber{y_saber}.jpg'

for i in range(100):
    x_saber = f'Saber{y_saber}.jpg'
    image = Image.open(x_saber)
    #print(image.size)
    newimg = np.array(Image.open(x_saber).resize((200,200)))
    
    if i == 0:
        x_train = np.reshape(newimg,(1,-1))
        x_train = x_train / 255
        x_train = x_train.tolist()
        x_train = np.array(x_train).astype('float32')
        #print(x_train.shape)
        y_train =  np.array([1])
    else:
        newn = np.reshape(newimg,(1,-1))
        newn = newn / 255
        newn = newn.tolist()
        x_train = np.append(x_train,newn)
        #print(x_train.shape)
        y_train = np.append(y_train,(0))
    #pyplot.imshow(newimg)
    #pyplot.show()   
    y_saber = y_saber + 1

y_rin = 1
x_rin = f'Rin{y_rin}.jpg'

for i in range(100):
    x_rin = f'Rin{y_rin}.jpg'
    image = Image.open(x_rin)
    #print(image.size)
    newimg = np.array(Image.open(x_rin).resize((200,200)))
    
    newn = np.reshape(newimg,(1,-1))
    newn = newn / 255
    newn = newn.tolist()
    
    x_train = np.append(x_train,newn)
    #print(x_train.shape)
    y_train = np.append(y_train,(1))
    
    # pyplot.imshow(newimg)
    # pyplot.show()
    y_rin = y_rin + 1

y_sakura = 1
x_sakura = f'Sakura{y_sakura}.jpg'

for i in range(100):
    x_sakura = f'Sakura{y_sakura}.jpg'
    image = Image.open(x_sakura)
    #print(image.size)
    newimg = np.array(Image.open(x_sakura).resize((200,200)))
    
    newn = np.reshape(newimg,(1,-1))
    newn = newn / 255
    newn = newn.tolist()
    
    x_train = np.append(x_train,newn)
    y_train = np.append(y_train,(2))
    #print(x_train.shape)
    # pyplot.imshow(newimg)
    # pyplot.show()
    y_sakura = y_sakura + 1

x_train = x_train.reshape(-1,120000)
#print(y_train)
#print(y_train.shape)
#print(x_train.shape)

y_saber = 7
x_saber = f'test{y_saber}.jpg'

for i in range(3):
    x_saber = f'test{y_saber}.jpg'
    image = Image.open(x_saber)
    #print(image.size)
    newimg = np.array(Image.open(x_saber).resize((200,200)))
    
    if i == 0:
        x_test = np.reshape(newimg,(1,-1))
        x_test = x_test / 255
        x_test = x_test.tolist()
        x_test = np.array(x_test).astype('float32')
        #print(x_test.shape)
        y_test =  np.array([1])
    else:
        newn = np.reshape(newimg,(1,-1))
        newn = newn / 255
        newn = newn.tolist()
        x_test = np.append(x_test,newn)
        #print(x_test.shape)
        y_test = np.append(y_test,(0))
    pyplot.imshow(newimg)
    #pyplot.show()   
    y_saber = y_saber + 1
#print(y_test)

y_saber = 1
x_saber = f'test{y_saber}.jpg'

for i in range(3):
    x_sakura = f'Sakura{y_saber}.jpg'
    image = Image.open(x_sakura)
    #print(image.size)
    newimg = np.array(Image.open(x_sakura).resize((200,200)))
    
    newn = np.reshape(newimg,(1,-1))
    newn = newn / 255
    newn = newn.tolist()
    
    x_test = np.append(x_test,newn)
    y_test = np.append(y_test,(2))
    #print(x_test.shape)
    pyplot.imshow(newimg)
    #pyplot.show()
    y_saber = y_saber + 1
#print(y_test)
#print(y_test.shape)

y_saber = 4
x_saber = f'test{y_saber}.jpg'

for i in range(3):
    x_rin = f'test{y_saber}.jpg'
    image = Image.open(x_rin)
    #print(image.size)
    newimg = np.array(Image.open(x_rin).resize((200,200)))
    
    newn = np.reshape(newimg,(1,-1))
    newn = newn / 255
    newn = newn.tolist()
    
    x_test = np.append(x_test,newn)
    y_test = np.append(y_test,(1))
    #print(x_test.shape)
    pyplot.imshow(newimg)
    #pyplot.show()
    y_saber = y_saber + 1
#print(y_test)
#print(y_test.shape)
#print(x_test.shape)


y_saber = 1
x_saber = f'testsaber{y_saber}.jpg'

for i in range(10):
    x_sakura = f'testsaber{y_saber}.jpg'
    image = Image.open(x_sakura)
    #print(image.size)
    newimg = np.array(Image.open(x_sakura).resize((200,200)))
    
    newn = np.reshape(newimg,(1,-1))
    newn = newn / 255
    newn = newn.tolist()
    
    x_test = np.append(x_test,newn)
    y_test = np.append(y_test,(0))
    #print(x_test.shape)
    pyplot.imshow(newimg)
    #pyplot.show()
    y_saber = y_saber + 1

y_saber = 1
x_saber = f'testrin{y_saber}.jpg'

for i in range(10):
    x_sakura = f'testrin{y_saber}.jpg'
    image = Image.open(x_sakura)
    #print(image.size)
    newimg = np.array(Image.open(x_sakura).resize((200,200)))
    
    newn = np.reshape(newimg,(1,-1))
    newn = newn / 255
    newn = newn.tolist()
    
    x_test = np.append(x_test,newn)
    y_test = np.append(y_test,(1))
    #print(x_test.shape)
    pyplot.imshow(newimg)
    #pyplot.show()
    y_saber = y_saber + 1

y_saber = 1
x_saber = f'testsakura{y_saber}.jpg'

for i in range(10):
    x_sakura = f'testsakura{y_saber}.jpg'
    image = Image.open(x_sakura)
    #print(image.size)
    newimg = np.array(Image.open(x_sakura).resize((200,200)))
    
    newn = np.reshape(newimg,(1,-1))
    newn = newn / 255
    newn = newn.tolist()
    
    x_test = np.append(x_test,newn)
    y_test = np.append(y_test,(2))
    #print(x_test.shape)
    pyplot.imshow(newimg)
    #pyplot.show()
    y_saber = y_saber + 1

import tensorflow.compat.v1 as tf
import keras
from keras.backend.tensorflow_backend import set_session
from keras.utils import to_categorical

def encode(data):
    #print('Shape of data (BEFORE encode): %s' % str(data.shape))
    encoded = to_categorical(data)
    #print('Shape of data (AFTER  encode): %s\n' % str(encoded.shape))
    return encoded
y_train =encode(y_train)
y_test = encode(y_test)
#print(y_test)


x_test = x_test.reshape(-1,200,200,3)
x_train = x_train.reshape(-1,200,200,3)
print(x_train.shape)
print(x_test.shape)

# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_virtual_device_configuration(gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1000)])

session = tf.Session()
tf.compat.v1.keras.backend.set_session(session)

from tensorflow.keras.layers import InputSpec
from keras.models import Sequential
from keras.layers import Dense
from keras import activations
from keras import optimizers
from keras import backend as K
from keras import *

def swish(x):
    return K.sigmoid(x) * x
 
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation=swish, input_shape=(200, 200, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation=swish))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation=swish))
model.add(layers.Flatten())
#model.add(Dense(15,input_dim = 120000,activation = swish))
model.add((Dense(3,activation='softmax')))

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss = 'categorical_crossentropy',optimizer='adam',metrics =['accuracy'])

model.fit(x_train,y_train,batch_size = 128,epochs = 100, verbose = 1,validation_data=(x_test,y_test))

score  = model.evaluate(x_test,y_test)
print('accuracy= ',score[1])

model.save('Fate_face_regc_V1.1')