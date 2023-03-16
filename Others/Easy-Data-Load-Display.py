#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
import random
import numpy as np
import glob
import cv2
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


import warnings
warnings.filterwarnings('ignore')
random.seed(23)


# # For segmentation

# In[5]:


#Load data

image_path = "Forestaeria/images/*.jpg"
mask_path = "Forestaeria/masks/*.jpg"

image_names = sorted(glob.glob(image_path), key=lambda x: x.split('.')[0])
mask_names = sorted(glob.glob(mask_path), key=lambda x: x.split('.')[0])


# In[6]:


#resizing images 
H=128
W=128
CH=3
nepochs = 25


# In[7]:


def data_loads(image_names,mask_names):
    X = []
    Y = []
    
    #images
    for image in image_names:
        img = cv2.imread(image, -1)
        img = cv2.resize(img,( H, W))
        X.append(img)
        
    X = np.array(X)
    X=X/255.0

#masks
    for mask in mask_names:
        msk = cv2.imread(mask, 0)
        msk = cv2.resize(msk, (H, W))
        Y.append(msk)
    
    Y = np.array(Y)
    Y=Y/255.0
    
    return X, Y


# In[35]:


## splitting the image into train and test

X, y =  data_loads(image_names,mask_names)
 
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1, random_state=23)

#Dimention
print("X_train shape:",X_train.shape)
print("X_val shape:",X_test.shape)
print("Y_train shape:",y_train.shape)
print("Y_val shape:",y_test.shape)


# In[9]:


#Display

def samples(X_test, y_test):
    figure, axes = plt.subplots(8,2, figsize=(30,30))
    
    for i in range(0,8):
        rand_num = random.randint(0,400)
        original_img = X_test[rand_num]
        axes[i,0].imshow(original_img)
        axes[i,0].title.set_text(' Image')
        
        original_mask = y_test[rand_num]
        axes[i,1].imshow(original_mask)
        axes[i,1].title.set_text('Mask')
        
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    plt.show()


# In[10]:


samples(X_test, y_test)


# # For classification 

# In[16]:


# Create an ImageDataGenerator
nbatch_size = 32
train_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
    directory = 'Forestaeria',
    target_size = (H, W),
    color_mode = 'grayscale',
    class_mode = 'binary',#catagorical for  multiclassification
    batch_size = nbatch_size
)

# Show class 
print("Class indices:", train_generator.class_indices)
print("\n")


# In[17]:


# Create a training and testing set
Xx, Yy = train_generator.next()
Xx_train, Xx_test, Yy_train, Yy_test = train_test_split(Xx, Yy, test_size =0.1, random_state=42)


# In[36]:


#normalization

Xx_train= Xx_train/255.0
Xx_test= Xx_test/255.0

Xx_train = Xx_train.reshape(-1,Xx_train.shape[1],Xx_train.shape[2],1)
Xx_test = Xx_test.reshape(-1,Xx_train.shape[1],Xx_train.shape[2],1)
Yy_train = Yy_train.reshape(Yy_train.shape[0],1)
Yy_testl = Yy_test.reshape(Yy_test.shape[0],1)

#Dimention
print("X_train shape:",Xx_train.shape)
print("X_val shape:",Xx_test.shape)
print("Y_train shape:",Yy_train.shape)
print("Y_val shape:",Yy_test.shape)



# In[40]:


figure, axes = plt.subplots(8,2, figsize=(10,10))
indexes=[]
for i in range(9):
    index=random.randint(0,10)    
    plt.subplot(3,3,i+1)
    plt.imshow(Xx_train[index])
    plt.axis("off")

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()


# In[ ]:




