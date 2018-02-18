from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)
from keras.preprocessing.image import ImageDataGenerator
size = 227
epochs = 50

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
# create the base pre-trained model
base_model = ResNet50(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(5, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics = ['accuracy'])

# train the model on the new data for a few epochs
model.fit_generator(X_train,
                       steps_per_epoch = 2000,
                       epochs = epochs,
                       validation_data = X_test,
                       validation_steps = 800 )

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:45]:
   layer.trainable = False
for layer in model.layers[45:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', accuracy = ['accuracy'])

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
model.fit_generator(X_train,
                       steps_per_epoch = 2000,
                       epochs = epochs,
                       validation_data = X_test,
                       validation_steps = 800)
