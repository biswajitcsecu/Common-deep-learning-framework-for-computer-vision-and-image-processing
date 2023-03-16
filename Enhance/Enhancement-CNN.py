#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import random
import numpy as np
import glob
import cv2
from glob import glob
import scipy
import cv2 as cv
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.layers import Input, Conv2D, Conv2DTranspose, Concatenate
from keras.applications.vgg19 import VGG19
from keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, UpSampling2D 
from tensorflow.keras.layers import Conv2DTranspose, Reshape, Dropout, concatenate, Concatenate, multiply
from tensorflow.keras.layers import Lambda, Activation, subtract, Flatten, Dense,add, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import imageio
from keras import backend as K
from keras.models import Model
from keras.utils import plot_model
from scipy import misc
from glob import glob
import tensorflow as tf
import numpy as np
import scipy
import platform
import keras
import os
import random

import warnings
warnings.filterwarnings('ignore')
random.seed(23)


# In[18]:


#Load data

imageh_path = "Brighten/high/*.png"
imagel_path = "Brighten/low/*.png"

imageh_path = sorted(glob(imageh_path), key=lambda x: x.split('.')[0])
imagel_path = sorted(glob(imagel_path), key=lambda x: x.split('.')[0])


# In[19]:


#resizing images 
H=128
W=128
CH=3

def data_loads(imageh_names,imagel_names):
    X = []
    Y = []
    
    #High images
    for image in imageh_names:
        imgh = cv2.imread(image, 1)
        imgh = cv2.resize(imgh,( H, W))
        X.append(imgh)
    #normalization        
    X = np.array(X)
    X=X/255.0

    #Low image
    for imgl in imagel_names:
        imgl = cv2.imread(imgl, 1)
        imgl = cv2.resize(imgl, (H, W))
        Y.append(imgl)
    #normalization    
    Y = np.array(Y)
    Y=Y/255.0
    
    return X, Y


# In[22]:


## splitting the image into train and test

X, y =  data_loads(imageh_path,imagel_path)
 
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1, random_state=23)

#Dimention
print("X_train shape:",X_train.shape)
print("X_val shape:",X_test.shape)
print("Y_train shape:",y_train.shape)
print("Y_val shape:",y_test.shape)


# In[23]:


#Display
print('<=----------Low-----------------=>') 
figure, axes = plt.subplots(8,2, figsize=(20,20))
for i in range(9):
        index=random.randint(0,10)    
        plt.subplot(3,3,i+1)
        plt.imshow(y_test[index])
        plt.axis("off")        

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()
print('<=---------------------------=>\n')   

print('<=----------High---------------=>') 
figure, axes = plt.subplots(8,2, figsize=(20,20))
indexes=[]    
for i in range(9):
        index= random.randint(0,40)    
        plt.subplot(3,3,i+1)
        plt.imshow(X_test[index])
        plt.axis("off")        
plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()


# In[24]:


K.clear_session()
def InstantiateModel(in_):
    
    model_1 = Conv2D(16,(3,3), activation='relu',padding='same',strides=1)(in_)
    model_1 = Conv2D(32,(3,3), activation='relu',padding='same',strides=1)(model_1)
    model_1 = Conv2D(64,(2,2), activation='relu',padding='same',strides=1)(model_1)
    
    model_2 = Conv2D(32,(3,3), activation='relu',padding='same',strides=1)(in_)
    model_2 = Conv2D(64,(2,2), activation='relu',padding='same',strides=1)(model_2)
    
    model_2_0 = Conv2D(64,(2,2), activation='relu',padding='same',strides=1)(model_2)
    
    model_add = add([model_1,model_2,model_2_0])
    
    model_3 = Conv2D(64,(3,3), activation='relu',padding='same',strides=1)(model_add)
    model_3 = Conv2D(32,(3,3), activation='relu',padding='same',strides=1)(model_3)
    model_3 = Conv2D(16,(2,2), activation='relu',padding='same',strides=1)(model_3)
    
    model_3_1 = Conv2D(32,(3,3), activation='relu',padding='same',strides=1)(model_add)
    model_3_1 = Conv2D(16,(2,2), activation='relu',padding='same',strides=1)(model_3_1)
    
    model_3_2 = Conv2D(16,(2,2), activation='relu',padding='same',strides=1)(model_add)
    
    model_add_2 = add([model_3_1,model_3_2,model_3])
    
    model_4 = Conv2D(16,(3,3), activation='relu',padding='same',strides=1)(model_add_2)
    model_4_1 = Conv2D(16,(3,3), activation='relu',padding='same',strides=1)(model_add)
    #Extension
    model_add_3 = add([model_4_1,model_add_2,model_4])
    
    model_5 = Conv2D(16,(3,3), activation='relu',padding='same',strides=1)(model_add_3)
    model_5 = Conv2D(16,(2,2), activation='relu',padding='same',strides=1)(model_add_3)
    
    model_5 = Conv2D(3,(3,3), activation='relu',padding='same',strides=1)(model_5)
    
    return model_5


# In[25]:


Input_Sample = Input(shape=(H, W, CH))
Output = InstantiateModel(Input_Sample)
model = Model(inputs=Input_Sample, outputs=Output)
model.compile(optimizer="adam", loss='mean_squared_error')
model.summary()


# In[26]:


from keras.utils.vis_utils import plot_model
plot_model(model,to_file='model_.png',show_shapes=True, show_layer_names=True)
from IPython.display import Image
Image(retina=True, filename='model_.png')


# In[27]:


def GenerateInputs(X,y):
    for i in range(len(X)):
        X_input = X[i].reshape(1,H,W,CH)
        y_input = y[i].reshape(1,H,W,CH)
        yield (X_input,y_input)        


# In[33]:


nbatch_size=32
nsteps_per_epoch=(len(imagel_path)//nbatch_size)
nepochs=30
history=model.fit(GenerateInputs(X_train,y_train),
                             epochs=nepochs,verbose=1,
                             steps_per_epoch=nsteps_per_epoch,shuffle=True,
                             use_multiprocessing =True)


# In[34]:


def ExtractTestInput(ImagePath):
    img = cv.imread(ImagePath)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img,(H,H))
    img1 = img.reshape(1,H,W,CH)
    return img1

def image_predict(Image_test2):
    plt.figure(figsize=(30,30))
    plt.subplot(5,5,1)
    img_1 = cv.imread(Image_test2)
    img_1 = cv.cvtColor(img_1, cv.COLOR_BGR2RGB)
    img_1 = cv.resize(img_1, (H, W))
    plt.title("Ground Truth",fontsize=20)
    plt.imshow(img_1)
    
    plt.subplot(5,5,1+1)
    img_ = ExtractTestInput(Image_test2)
    Prediction = model.predict(img_)
    img_ = img_.reshape(H, W, CH)
    plt.title("Low Light Image",fontsize=20)
    plt.imshow(img_)
    
    plt.subplot(5,5,1+2)
    Prediction = Prediction.reshape(H, W, CH)
    img_[:,:,:] = Prediction[:,:,:]
    plt.title("Enhanced Image",fontsize=20)
    plt.imshow(img_)


# In[35]:


TestPath="Brighten/low/"
Image_test=TestPath+"r0b7b1156t.png"
image_predict(Image_test)


# In[ ]:




