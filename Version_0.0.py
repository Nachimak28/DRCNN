#importing essential libraries
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
#import pydot
#from IPython.display import SVG
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
from tkinter import filedialog
from tkinter import *
tic = time.time()


a = input('Already trained? Y/n')


size = 128         #size of input image (128,128)
epochs = 2        #specifying number of epochs

if a == 'n':
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
    '''
    train_labels = np.ones((m_train,), dtype = 'int64')     #declaring a numpy array of '1's of size = m_train
    train_labels[:1344] = 0                                 #assigning actual labels of this numpy array according to subdirectory contents
    train_labels[1344:2275] = 1
    train_labels[2275:3555] = 2
    train_labels[3555:3965] = 3
    train_labels[3965:] = 4

    Y_train = keras.utils.to_categorical(train_labels, num_classes = 5)     #one-hot encoding

    test_labels = np.ones((m_test,), dtype = 'int64')       #declaring a numpy array of '1's of size = m_test
    test_labels[:337] = 0                                   #assigning actual labels of this numpy array according to subdirectory contents
    test_labels[337:568] = 1
    test_labels[568:896] = 2
    test_labels[896:1006] = 3
    test_labels[1006:] = 4

    Y_test = keras.utils.to_categorical(test_labels, num_classes = 5)       #one-hot encoding
    '''
    input_shape = (size, size, 3)

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


    #creating model by calling the model
    dr_model = DR_model(input_shape)

    #Compile the model. Specify the optimizer, loss function and desirable metrics
    dr_model.compile('adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

    dr_model.fit_generator(X_train,
                           steps_per_epoch = 1,
                           epochs = epochs,
                           validation_data = X_test,
                           validation_steps = 2 )

    dr_model.save('Version_0_model.h5')
    dr_model.save_weights('Version_0_model_weights.h5')
    print('Weights and model saved')

else:
    dr_model = load_model('Version_0_model.h5')
    dr_model.load_weights('Version_0_model_weights.h5')
    print('Weights and model loaded')

root = Tk()
root.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
print (root.filename)

img_path = root.filename
img = image.load_img(img_path, target_size = (size,size))
imshow(img)

x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)
x = preprocess_input(x)

print(dr_model.predict(x))
'''
plot_model(dr_model, to_file='DRModel.png')
#SVG(model_to_dot(dr_model).create(prog='dot', format='svg'))
'''

toc = time.time()
print("Time taken= " + str((toc-tic)) + "s")
