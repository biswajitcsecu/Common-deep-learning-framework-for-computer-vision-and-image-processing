#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Libraries
import pandas as pd
import os
import shutil
import random
import itertools
import numpy as np
import pathlib
from matplotlib import pyplot as plt
import cv2
import skimage
from skimage.util import random_noise
from tqdm.notebook import tqdm
import random
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score
import keract
import tensorflow as tf
import matplotlib as mpl
from keras import backend
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.applications import imagenet_utils
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet import decode_predictions, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from math import log10,sqrt

import warnings
warnings.filterwarnings('ignore')


# In[19]:


H=128
W=128
CH=1

root_dir='Polyp/train/images/'
train_images=sorted(os.listdir(root_dir))

train_image=[]
for im in train_images:
    img=image.load_img(root_dir+im,target_size=(H,W),color_mode='grayscale')
    img=image.img_to_array(img)
    img=img/255
    train_image.append(img)

train_df=np.array(train_image)
     


# In[20]:


#Subplotting images
def plot_img(dataset):
    f,ax=plt.subplots(1,5)
    f.set_size_inches(40,20)
    for i in range(5,10):
        ax[i-5].imshow(dataset[i].reshape(H,W), cmap='gray')
    plt.show()


# In[21]:


#Adding gaussian noise with 0.05 factor
def add_noise(image):
    row,col,ch=image.shape
    mean=0
    sigma=1
    gauss=np.random.normal(mean,sigma,(row,col,ch))
    gauss=gauss.reshape(row,col,ch)
    noisy=image+gauss*0.05
    
    return noisy
     


# In[22]:


#noised image
noised_df=[]
for img in train_df:
    noisy=add_noise(img)
    noised_df.append(noisy)

noised_df=np.array(noised_df)
     


# In[23]:


plot_img(train_df)


# In[24]:


plot_img(noised_df)


# In[25]:


#Slicing
xnoised=noised_df[:500]
xtest=noised_df[500:]     


# In[26]:


def autoencoder():
    input_img=Input(shape=(H,W,CH),name='image_input')
    #enoder 
    x = Conv2D(64, (3,3), activation='relu', padding='same', name='Conv1')(input_img)
    x = MaxPooling2D((2,2), padding='same', name='pool1')(x)
    x = Conv2D(64, (3,3), activation='relu', padding='same', name='Conv2')(x)
    x = MaxPooling2D((2,2), padding='same', name='pool2')(x) 
    
    #decoder
    x = Conv2D(64, (3,3), activation='relu', padding='same', name='Conv3')(x)
    x = UpSampling2D((2,2), name='upsample1')(x)
    x = Conv2D(64, (3,3), activation='relu', padding='same', name='Conv4')(x)
    x = UpSampling2D((2,2), name='upsample2')(x)
    x = Conv2D(1, (3,3), activation='sigmoid', padding='same', name='Conv5')(x)
    
    #model
    autoencoder = Model(inputs=input_img, outputs=x)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    
    return autoencoder


# In[27]:


model= autoencoder()
model.summary()


# In[ ]:


nepochs=50
nbatch_size=64

with tf.device('/device:CPU:0'):
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
    history=model.fit(xnoised, xnoised, epochs=nepochs, batch_size=nbatch_size, 
                      validation_data=(xtest, xtest), callbacks=[early_stopping],
                     verbose=1,
                     use_multiprocessing=True)    
    
    


# In[ ]:


xtrain= train_df[100:]


# In[ ]:


pred= model.predict(xtest[:5])
def plot_predictions(y_true, y_pred):    
    f, ax = plt.subplots(4, 5)
    f.set_size_inches(10.5,7.5)
    for i in range(5):
        ax[0][i].imshow(np.reshape(xtrain[i], (H,W)), aspect='auto', cmap='gray')
        ax[1][i].imshow(np.reshape(y_true[i], (H,W)), aspect='auto', cmap='gray')
        ax[2][i].imshow(np.reshape(y_pred[i], (H,W)), aspect='auto', cmap='gray')
        ax[3][i].imshow(cv2.medianBlur(xtrain[i], (5)), aspect='auto', cmap='gray')
    plt.tight_layout()
plot_predictions(xtest[:5], pred[:5])


# In[ ]:


#instant method
median_blur = cv2.medianBlur(xtrain[0], (5))
gaussian_blur=cv2.GaussianBlur(xtrain[0],(5,5),0)
average_blur=cv2.blur(xtrain[0],(5,5))
bilateral_filter=cv2.bilateralFilter(xtrain[0],9,75,75)

f,ax=plt.subplots(1,5)
f.set_size_inches(40,20)
ax[0].imshow(pred[0].reshape(64,64), cmap='gray')
ax[0].set_title('Autoencoder Image')
ax[1].imshow(median_blur,cmap='gray')
ax[1].set_title('Median Filter')
ax[2].imshow(gaussian_blur,cmap='gray')
ax[2].set_title('Gaussian Filter')
ax[3].imshow(average_blur,cmap='gray')
ax[3].set_title('Average Filter')
ax[4].imshow(bilateral_filter,cmap='gray')
ax[4].set_title('Bilateral Filter')





# In[ ]:


#PSNR----------

def PSNR(original, denoised): 
    mse = np.mean((original - denoised) ** 2) 
    if(mse == 0): 
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr 

  
value1 = PSNR(xtest[0], median_blur)
value2 = PSNR(xtest[0], pred[0])
value3 = PSNR(xtest[0], gaussian_blur)
value4 = PSNR(xtest[0], average_blur)
value5 = PSNR(xtest[0], bilateral_filter)

print("PSNR values")
print(f"Autoencoder Image : {value2} dB")
print(f"Median Filter Image : {value1} dB")
print(f"Gaussian Filter Image : {value3} dB")
print(f"Average Filter Image : {value4} dB")
print(f"Bilateral Filter Image : {value5} dB")
     


# In[ ]:




