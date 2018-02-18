#importing essential libraries
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Dropout
from keras.models import Model, load_model
from keras import optimizers
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
#from resnets_utils import *
from keras.initializers import glorot_uniform
#import scipy.misc
from matplotlib.pyplot import imshow
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)
#import os
#from scipy.misc import imread, imsave, imresize
#from sklearn.utils import shuffle
#from sklearn.cross_validation import train_test_split
#from sklearn import preprocessing
import glob
import scipy.misc
import time
tic = time.time()


size = 227         #size of input image (w,h)
epochs = 50        #specifying number of epochs

train_data = ImageDataGenerator(rescale = 1./255, horizontal_flip = True, vertical_flip = True,samplewise_std_normalization=True)     #preprocessing and real tiime data augmentation of training set images

test_data = ImageDataGenerator(rescale = 1./255, horizontal_flip = True, vertical_flip = True,samplewise_std_normalization=True)       #preprocessing and real tiime data augmentation of test set images

X_train = train_data.flow_from_directory('C:/Users/nachiket/Desktop/SEM_8/BE_project/train_data',
                                         target_size = (size, size),
                                         batch_size = 32,
                                         class_mode = 'categorical')    #Creates a DirectoryIterator object  for getting images from the directory specified with images in sub directories 0,1,2,3,4 for train set 


X_test = test_data.flow_from_directory('C:/Users/nachiket/Desktop/SEM_8/BE_project/test_data',
                                       target_size = (size, size),
                                       batch_size = 32,
                                       class_mode = 'categorical')       #Creates a DirectoryIterator object  for getting images from the directory specified with images in sub directories 0,1,2,3,4 for train set 


m_train = 4310        #number of images in training set
m_test = 1093         #number of images in test set

input_shape = (size, size, 3)

##################################################################################################3
'''
LeNet Architecture

Convolution layer 1 = 20, filter size = (5,5), strides = (1,1)
Activation = tanh

MaxPooling 1 = filter size = (2,2), strides = (2,2)
Convolution layer 2 = 50, filter size = (5,5), strides = (1,1)
Activation = tanh

MaxPooling 2 = filter size = (2,2), strides = (2,2)

Flatten

FC 1 = 500
Activation = tanh


FC 2 = 5
Activation = Softmax
'''
##########################################################################################################


def DR_LeNet(input_shape):

    X_input = Input(input_shape)

    X = Conv2D(20, (5,5), strides = (1,1), name = 'CONV1', kernel_initializer = glorot_uniform(seed=0))(X_input)
    X = Activation('tanh')(X)
    
    X = MaxPooling2D((2,2), strides = (2,2), name = 'MAXPOOL1')(X)
    X = Conv2D(50, (5,5), strides = (1,1), name = 'CONV2', kernel_initializer = glorot_uniform(seed=0))(X)
    X = Activation('tanh')(X)

    X = MaxPooling2D((2,2), strides = (2,2), name = 'MAXPOOL2')(X)
    
    X = Flatten()(X)

    X = Dense(500, activation = 'tanh', name = 'FC1')(X)

    X = Dense(5, activation = 'softmax', name = 'FC2')(X)

    model = Model(inputs = X_input, outputs = X, name = 'DR_LeNet')

    return model


#creating model by calling the model
dr_model = DR_LeNet(input_shape)

#Compile the model. Specify the optimizer, loss function and desirable metrics
#adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.0)   #setting up the parameters of adam optimizer i.e. learning rate
sgd = optimizers.SGD(lr=0.01, clipvalue=0.5)
dr_model.compile(sgd, loss = 'categorical_crossentropy', metrics = ['accuracy'])

dr_model.fit_generator(X_train,
                       steps_per_epoch = 2000,
                       epochs = epochs,
                       validation_data = X_test,
                       validation_steps = 800 )



toc = time.time()
print("Time taken= " + str((toc-tic)) + "s")
