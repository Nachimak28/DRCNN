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
#X_train = np.array(X_train)         #converting a DirectoryIterator object to numpy array for supplying to database

X_test = test_data.flow_from_directory('C:/Users/nachiket/Desktop/SEM_8/BE_project/test_data',
                                       target_size = (size, size),
                                       batch_size = 32,
                                       class_mode = 'categorical')       #Creates a DirectoryIterator object  for getting images from the directory specified with images in sub directories 0,1,2,3,4 for train set 
#X_test = np.array(X_test)         #converting a DirectoryIterator object to numpy array for supplying to database

m_train = 4310        #number of images in training set
m_test = 1093         #number of images in test set

input_shape = (size, size, 3)
'''
#below is the model description function
def DR_model(input_shape):

    #define input placeholder as a tensor with input_shape
    X_input = Input(input_shape)

    #Zero_padding: pads border of X_input with zeroes
    X = ZeroPadding2D((3,3))(X_input)

    #CONV -> BN -> RELU block applied to X
    X = Conv2D(32, (7,7), strides = (1,1), name = 'conv0')(X)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)

    #MAXPOOL
    X = MaxPooling2D((2,2), name = 'max_pool')(X)

    #FLATTEN X (means convert it to a vector) + FULLY CONNECTED
    X = Flatten()(X)
    X = Dense(5, activation = 'softmax', name = 'fc')(X)

    #Create model. This creates Keras model instance
    model = Model(inputs = X_input, outputs = X, name = 'DR_model')

    return model
'''
##################################################################################################3
'''
AlexNet architecture detail

Convolution layer 1 = 96, filter size = (11,11), strides = (4,4), padding = 0
Activation = ReLU

MaxPooling 1 = filter size = (3,3), strides = (2,2)
BatchNormalization 1
Convolution layer 2 = 256, filter size = (5,5), strides = (1,1), padding = 2
Activation = ReLU

MaxPooling 2 = filter size = (3,3), strides = (2,2)
BatchNormalization 2
Convolution layer 3 = 384, filter size = (3,3), strides = (1,1), padding = 1
Activation = ReLU

Convolution layer 4 = 384, filter size = (3,3), strides = (1,1), padding = 1
Activation = ReLU

Convolution layer 5 = 256, filter size = (3,3), strides = (1,1), padding = 1
Activation = ReLU

MaxPooling 3 = filter size = (3,3), strides = (2,2)

Flatten

FC 6 = 4096
Activation = ReLU
Dropout = p

FC 7 = 4096
Activation = ReLU
Dropout = p

FC 8 = 5
Activation = Softmax

'''
##########################################################################################################


def DR_AlexNet(input_shape):

    X_input = Input(input_shape)

    X = Conv2D(96, (11,11), strides = (4,4), padding= 'valid', name = 'CONV1', kernel_initializer = glorot_uniform(seed=0))(X_input)
    X = Activation('relu')(X)
    
    X = MaxPooling2D((3,3), strides = (2,2), name = 'MAXPOOL1')(X)
    X = BatchNormalization(axis = 3, name = 'BN1')(X)
    X = Conv2D(256, (5,5), strides = (1,1), name = 'CONV2', kernel_initializer = glorot_uniform(seed=0))(X)
    X = ZeroPadding2D((2,2))(X)
    X = Activation('relu')(X)

    X = MaxPooling2D((3,3), strides = (2,2), name = 'MAXPOOL2')(X)
    X = BatchNormalization(axis = 3, name = 'BN2')(X)
    X = Conv2D(384, (3,3), strides = (1,1), name = 'CONV3', kernel_initializer = glorot_uniform(seed=0))(X)
    X = ZeroPadding2D((1,1))(X)
    X = Activation('relu')(X)

    X = Conv2D(384, (3,3), strides = (1,1), name = 'CONV4', kernel_initializer = glorot_uniform(seed=0))(X)
    X = ZeroPadding2D((1,1))(X)
    X = Activation('relu')(X)

    X = Conv2D(256, (3,3), strides = (1,1), name = 'CONV5', kernel_initializer = glorot_uniform(seed=0))(X)
    X = ZeroPadding2D((1,1))(X)
    X = Activation('relu')(X)

    X = MaxPooling2D((3,3), strides = (2,2), name = 'MAXPOOL3')(X)

    X = Flatten()(X)

    X = Dense(4096, activation = 'relu', name = 'FC6')(X)
    X = Dropout(0.25)(X)

    X = Dense(4096, activation = 'relu', name = 'FC7')(X)
    X = Dropout(0.25)(X)

    X = Dense(5, activation = 'softmax', name = 'FC8')(X)

    model = Model(inputs = X_input, outputs = X, name = 'DR_AlexNet')

    return model


#creating model by calling the model
dr_model = DR_AlexNet(input_shape)

#Compile the model. Specify the optimizer, loss function and desirable metrics
adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.0)   #setting up the parameters of adam optimizer i.e. learning rate
dr_model.compile('adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

dr_model.fit_generator(X_train,
                       steps_per_epoch = 2000,
                       epochs = epochs,
                       validation_data = X_test,
                       validation_steps = 800 )



toc = time.time()
print("Time taken= " + str((toc-tic)) + "s")
