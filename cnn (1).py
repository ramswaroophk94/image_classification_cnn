# Convolutional Neural Network

# Installing Keras
# Enter the following command in a terminal (or anaconda prompt for Windows users): conda install -c conda-forge keras

# Part 1 - Building the CNN
import pip
!pip install keras

!pip install tensorflow

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')
#Train the model
history=classifier.fit_generator(training_set,
                         steps_per_epoch = 80,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 20)

#summary of the model
classifier.summary()

#Compile the model
classifier.compile(loss='binary_crossentropy',
                   optimizer='rmsprop',
                   metrics=['accuracy'])
     

#Visualize training results
#Now visualize the results after training the network.
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs=25
epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


#Testing the model with with diffrent input images
#input : shoe
from PIL import Image
 # load the image
image1 = Image.open('shoes1.jpg')
# summarize some details about the image
print(image1.format)
print(image1.mode)
print(image1.size)
# show the image
image1.show()

img_pred= image.load_img('shoes1.jpg', target_size = (64, 64))
img_pred=image.img_to_array(img_pred)
img_pred=np.expand_dims(img_pred,axis=0)
result=classifier.predict(img_pred)
print(result)
if result[0][0] == 0:
    prediction="Handbags"
else:
    prediction="Shoes"
    
print(prediction) 


#input : handbag
image2 = Image.open('handbag1.jpg')
# summarize some details about the image
print(image2.format)
print(image2.mode)
print(image2.size)
# show the image
image2.show()
img_pred= image.load_img('handbag1.jpg', target_size = (64, 64))
img_pred=image.img_to_array(img_pred)
img_pred=np.expand_dims(img_pred,axis=0)


result=classifier.predict(img_pred)
print(result)
if result[0][0] == 0:
    prediction="Handbags"
else:
    prediction="Shoes"
    
print(prediction) 

#input : both shoe and handbag
image3 = Image.open('puma_both.jpg')
# summarize some details about the image
print(image3.format)
print(image3.mode)
print(image3.size)
# show the image
image3.show()
img_pred= image.load_img('puma_both.jpg', target_size = (64, 64))
img_pred=image.img_to_array(img_pred)
img_pred=np.expand_dims(img_pred,axis=0)


result=classifier.predict(img_pred)
print(result)
if result[0][0] == 0:
    prediction="Handbags"
else:
    prediction="Shoes"
    
print(prediction) 
