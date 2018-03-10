from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model, load_model
from keras import optimizers
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers 
import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import preprocess_input
from sklearn.metrics import roc_auc_score
from tkinter import filedialog
from tkinter import *
import numpy as np
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
import time
tic = time.time()
size = 224
#epochs = 50
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

a = input("Already trained the model? Y/n: ")

if a == 'n':
    train_data = ImageDataGenerator(rescale = 1./255, horizontal_flip = True, vertical_flip = True,samplewise_std_normalization=True, shear_range = 0.2, zoom_range = 0.1)     #preprocessing and real tiime data augmentation of training set images

    test_data = ImageDataGenerator(rescale = 1./255, horizontal_flip = True, vertical_flip = True,samplewise_std_normalization=True, shear_range = 0.2, zoom_range = 0.1)       #preprocessing and real tiime data augmentation of test set images

    X_train = train_data.flow_from_directory('C:/Users/nachiket/Desktop/SEM_8/BE_project/final_dataset/Final_dataset_train',
                                             target_size = (size, size),
                                             batch_size = 64,
                                             class_mode = 'categorical')    #Creates a DirectoryIterator object  for getting images from the directory specified with images in sub directories 0,1,2,3,4 for train set 


    X_test = test_data.flow_from_directory('C:/Users/nachiket/Desktop/SEM_8/BE_project/final_dataset/Final_dataset_test',
                                           target_size = (size, size),
                                           batch_size = 64,
                                           class_mode = 'categorical')       #Creates a DirectoryIterator object  for getting images from the directory specified with images in sub directories 0,1,2,3,4 for train set 


    #m_train = 4310        #number of images in training set
    #m_test = 1093         #number of images in test set

    #train_labels = np.ones((m_train,), dtype = 'int64')     #declaring a numpy array of '1's of size = m_train
    #train_labels[:1344] = 0                                 #assigning actual labels of this numpy array according to subdirectory contents
    #train_labels[1344:2275] = 1
    #train_labels[2275:3555] = 2
    #train_labels[3555:3965] = 3
    #train_labels[3965:] = 4

    #Y_train = to_categorical(train_labels, num_classes = 5)     #one-hot encoding

    #test_labels = np.ones((m_test,), dtype = 'int64')       #declaring a numpy array of '1's of size = m_test
    #test_labels[:337] = 0                                   #assigning actual labels of this numpy array according to subdirectory contents
    #test_labels[337:568] = 1
    #test_labels[568:896] = 2
    #test_labels[896:1006] = 3
    #test_labels[1006:] = 4

    #Y_test = to_categorical(test_labels, num_classes = 5)  

    input_shape = (size, size, 3)
    # create the base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu',kernel_regularizer=regularizers.l2(0.01))(x)
    x = Dropout(0.25)(x)


    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(5, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers

    # at this point, the top layers are well trained and we can start fine-tuning
    # convolutional layers from inception V3. We will freeze the bottom N layers
    # and train the remaining top layers.

    # let's visualize layer names and layer indices to see how many layers
    # we should freeze:
    for i, layer in enumerate(base_model.layers):
       print(i, layer.name)

    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 249 layers and unfreeze the rest:
    for layer in model.layers[:249]:
       layer.trainable = False
    for layer in model.layers[249:]:
       layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    #from keras.optimizers import ADAM
    model.compile(optimizer=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999), loss='categorical_crossentropy', metrics = ['accuracy'])

    # we train our model again (this time fine-tuning the top 2 inception blocks
    # alongside the top Dense layers
    model.fit_generator(X_train,
                           steps_per_epoch = (22468/64),
                           epochs = 30,
                           validation_data = X_test,
                           validation_steps = 40)

    #y_score = model.predict_proba(X_test)

    #roc_auc_score(Y_test, y_scores)

    model.save('Inception_retrained.h5')
    model.save_weights('Inception_retrained_weights.h5')

else:
    model = load_model('Inception_retrained.h5')
    model.load_weights('Inception_retrained_weights.h5')
    print('Weights and model loaded')

root = Tk()
root.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
print (root.filename)

img_path = root.filename
img = image.load_img(img_path, target_size = (size,size))
#imshow(img)

x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)
x = preprocess_input(x)

op = (model.predict(x))*100
print(np.round_(op,4))
print(type(model.predict(x)))
print(op.shape)
print('Catrgory 0: ' + str(op[0][0]))
print('Catrgory 1: ' + str(op[0][1]))
print('Catrgory 2: ' + str(op[0][2]))
print('Catrgory 3: ' + str(op[0][3]))
print('Catrgory 4: ' + str(op[0][4]))


plot_model(model, to_file='DRModelInception.png')
toc = time.time()
print("Time taken= " + str((toc-tic)) + "s")
