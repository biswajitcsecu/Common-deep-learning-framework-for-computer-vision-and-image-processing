#!/usr/bin/env python
# coding: utf-8

# In[217]:


import os
import random
import glob
import cv2
import numpy as np
import pandas as pd
import seaborn as sb
from tqdm import tqdm
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
from matplotlib import style
import keras
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


import warnings
warnings.filterwarnings('ignore')
random.seed(23)
plt.style.use('seaborn')


# # Preprocessing

# In[218]:


# images info
H=128
W=128
CH=3
nepochs = 25
nbatch_size = 16

#Labelling images
root_dir = "Lion_Cheetah/train/"
class_names = sorted(os.listdir(root_train_dir))
num_classes = len(class_names)
print('classes: ', class_names)


# In[219]:


#Image Batch
labels = []
images = []

for clidx in class_names:
    print(clidx, end=' ==> ')
    for img in os.listdir(root_dir + clidx):
        label = np.zeros(num_classes)
        label[class_names.index(clidx)] = 1
        labels.append(label)        
        image = np.asarray(cv2.resize(cv2.imread(root_dir + clidx + '/' + img, cv2.IMREAD_COLOR), (H, W)))
        images.append(image)
    print('done')

labels = np.asarray(labels)
images = np.asarray(images)

# Show class 
print(f'\n\nlabels shape: {labels.shape}')
print(f'images shape: {images.shape}')
print("\n")


# In[220]:


# Create a training and testing set
X_train, X_val, Y_train, Y_val = train_test_split(images, labels, test_size =0.2, random_state=42)

#dimention
print("X_train shape:",X_train.shape)
print("X_val shape:",X_val.shape)
print("Y_train shape:",Y_train.shape)
print("Y_val shape:",Y_val.shape)


# In[221]:


fig, axes = plt.subplots(8,2, figsize=(9,9))
fig.suptitle('Category')
indexes=[]
for i in range(9):
    index=random.randint(0,10)    
    plt.subplot(3,3,i+1)
    plt.imshow(X_train[index],cmap='gray')
    plt.axis("off")

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()


# # Augmentation Data

# In[222]:


# Create an ImageDataGenerator

training_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,    
    horizontal_flip=False,
    vertical_flip=False,
    preprocessing_function=None,
    data_format=None,
    interpolation_order=1,
    dtype=None,
    shear_range=0.1,
    fill_mode='nearest'
)

train_generator = training_datagen.flow(
    X_train,
    y=Y_train,
    batch_size=nbatch_size
)

val_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    fill_mode='nearest'
)

val_generator = val_datagen.flow(
    X_val, 
    y=Y_val,
    batch_size=nbatch_size
)


# In[223]:


img_size = (H, W, CH)

model = tf.keras.models.Sequential([
    #first convolution
    tf.keras.layers.Conv2D(8, (3,3), activation='relu', input_shape=img_size),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Conv2D(16, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

# Print the model summary
model.summary()



# # Compile model and set optimizer

# In[224]:


# Set the training parameters
model.compile(loss = 'categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(learning_rate = 0.001),
              metrics=['accuracy'])


# In[ ]:


# Train the model
history = model.fit(train_generator, epochs=nepochs, validation_data = val_generator, verbose=1, use_multiprocessing=True)


# # Evaluate Model   

# In[ ]:


#Evaluate Model
model.save('classifier.h5')
model.evaluate(val_generator)


# # Plot accuracy and loss

# In[ ]:


# Plot for accuracy and loss 
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Testing accuracy')
plt.title('Training and testing accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Testing Loss')
plt.title('Training and testing loss')
plt.legend()

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()


# In[ ]:


pred=model.predict(val_generator)
predictions=np.argmax(pred,axis=1)


# In[ ]:


classes= ['Cheetahs', 'Lions']
i=0
count=0
correct_class=[]
incorrect_class=[]

for i in range(len(Y_val)):
    if(np.argmax(Y_val[i])==predictions[i]):
        correct_class.append(i)
    if(len(correct_class)==8):
        break

for i in range(len(Y_val)):    
    if (np.argmax(Y_val[i])!=predictions[i]):        
        incorrect_class.append(i)
    if (len(incorrect_class)==8):
        break


# In[ ]:


#plot
count=0
fig,ax=plt.subplots(4,2)
fig.set_size_inches(15,15)

for i in range (4):
    for j in range (2):
        ax[i,j].imshow(X_val[correct_class[count]])
        ax[i,j].set_title("Predicted big-cat : "+ classes[predictions[correct_class[count]]] 
                          +"\n"+"Actual big-cat : "+ classes[np.argmax(Y_val[correct_class[count]])])
        plt.tight_layout()
        count+=1
plt.show()


# In[ ]:


count=0
fig,ax=plt.subplots(4,2)
fig.set_size_inches(15,15)
for i in range(4):
    for j in range(2):
        ax[i,j].imshow(X_val[incorrect_class[count]])
        ax[i,j].set_title("Predicted  big-cat : " + classes[predictions[incorrect_class[count]]] + 
                          "\n"+"Actual  big-cat : " +classes[np.argmax(Y_val[incorrect_class[count]])])
        plt.tight_layout()
        count+=1

