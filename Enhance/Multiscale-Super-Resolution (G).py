#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os 
import re 
from scipy import ndimage, misc 
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import img_to_array


from skimage.transform import resize, rescale
import matplotlib.pyplot as plt
import numpy as np
np. random. seed(0)
import cv2 as cv2

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense ,Conv2D,MaxPooling2D ,Dropout
from tensorflow.keras.layers import Conv2DTranspose, UpSampling2D, add ,concatenate
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.utils import plot_model
import tensorflow as tf

print(tf.__version__)


# In[3]:


def sorted_alphanumeric(data):  
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)',key)]
    return sorted(data,key = alphanum_key)


# In[15]:


# defining the size of the image
SIZE = 128

#high resolution image
high_img = []
path = 'LOL/train/high/'
files = os.listdir(path)
files = sorted_alphanumeric(files)

for i in tqdm(files):
        img = cv2.imread(path+i,1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (SIZE, SIZE))
        img = img.astype('float32') / 255.0
        high_img.append(img_to_array(img))

#low resolution image
low_img = []
path = 'LOL/train/low/'
files = os.listdir(path)
files = sorted_alphanumeric(files)

for i in tqdm(files): 
        img = cv2.imread(path + i,1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (SIZE, SIZE))
        img = img.astype('float32') / 255.0
        low_img.append(img_to_array(img)).
        
#image info
print('<=-------------Size------------=>')
print(f'Length of HR:',  len(high_img))
print('\n')
print(f'Length of LR:',  len(low_img))
print('<=-------------done------------=>')


# In[7]:


#visulization----------
for i in range(8):
    a = np.random.randint(0,100)
    plt.figure(figsize=(10,10))
    plt.subplot(1,2,1)
    plt.title('High Resolution Imge', color = 'green', fontsize = 20)
    plt.imshow(high_img[a])
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.title('low Resolution Image ', color = 'black', fontsize = 20)
    plt.imshow(low_img[a])
    plt.axis('off')


# In[17]:


#Data slicing--------
lr=300
ur=400
#train data
train_high_image = high_img[:lr]
train_low_image = low_img[:lr]
train_high_image = np.reshape(train_high_image,(len(train_high_image),SIZE,SIZE,3))
train_low_image = np.reshape(train_low_image,(len(train_low_image),SIZE,SIZE,3))

#valid data
validation_high_image = high_img[lr:ur]
validation_low_image = low_img[lr:ur]
validation_high_image= np.reshape(validation_high_image,(len(validation_high_image),SIZE,SIZE,3))
validation_low_image = np.reshape(validation_low_image,(len(validation_low_image),SIZE,SIZE,3))

#test
test_high_image = high_img[ur:]
test_low_image = low_img[ur:]
test_high_image= np.reshape(test_high_image,(len(test_high_image),SIZE,SIZE,3))
test_low_image = np.reshape(test_low_image,(len(test_low_image),SIZE,SIZE,3))

##info
print("Shape of training images:",train_high_image.shape)
print("Shape of validation images:",validation_high_image.shape)
print("Shape of test images:",test_high_image.shape)


# # Model Building

# In[25]:


#model parameters
H=128
W=128
CH=3
input_shape =Input(shape=(H,W,CH))
kernel_size = 3
dropout = 0.4
n_filters = 64


# In[21]:


def Upsample_block(x,ch=256, k_s=3, st=1):
    x = tf.keras.layers.Conv2D(ch,k_s, strides=(st,st), padding='same')(x)
    x = tf.nn.depth_to_space(x, 2)
    x = tf.keras.layers.LeakyReLU()(x)
    return x

# branch of Y network (L)
left_inputs = Input(shape=(H,W,CH))
x = left_inputs
filters = n_filters

for i in range(1):
    x = Conv2D(filters=filters,kernel_size=kernel_size,padding='same',activation='relu')(x)
    x = Dropout(dropout)(x)
    x = MaxPooling2D()(x)
    filters *= 2
    
#branch of Y network(R)
right_inputs = Input(shape=(H,W,CH))
y = right_inputs
filters = n_filters

for i in range(1):
    y = Conv2D(filters=filters,kernel_size=kernel_size,padding='same',activation='relu')(y)
    y = Dropout(dropout)(y)
    y = MaxPooling2D()(y)
    filters *= 2

y = add([x, y])
y=Upsample_block(y)
outputs=Conv2D (3,(3,3) , padding='same' ,activation='relu',activity_regularizer=regularizers.l1(10e-10))(y)
model= Model([left_inputs, right_inputs], outputs)


# In[22]:


model= Model([left_inputs, right_inputs], outputs)
model.summary()


# In[24]:


model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001), loss = 'mean_absolute_error',metrics = ['acc'])
plot_model(model, to_file='Multi_scale_learning.png', show_shapes=True)


# In[26]:


nbatch_size = 32
nepochs=20
model.fit([train_low_image, train_low_image],
          train_high_image,
          validation_data=([validation_low_image, validation_low_image], validation_high_image),
          epochs=nepochs,
          batch_size=nbatch_size,
          verbose=1,
          use_multiprocessing=True
         )


# In[27]:


#Metric

def PSNR(y_true,y_pred):
    mse=tf.reduce_mean( (y_true - y_pred) ** 2 )
    return 20 * log10(1 / (mse ** 0.5))

def log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def pixel_MSE(y_true,y_pred):
    return tf.reduce_mean( (y_true - y_pred) ** 2 )


# # Display predicted results

# In[28]:


#pridction

def plot_images(high,low,predicted):
    plt.figure(figsize=(15,15))
    plt.subplot(1,3,1)
    plt.title('High Image', color = 'green', fontsize = 20)
    plt.imshow(high)
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.title('Low Image ', color = 'black', fontsize = 20)
    plt.imshow(low)
    plt.axis('off')
    plt.subplot(1,3,3)
    plt.title('Predicted Image ', color = 'Red', fontsize = 20)
    plt.imshow(predicted)  
    plt.axis('off')
    plt.tight_layout()
    plt.show()

for i in range(10,20):
    SIZE=128
    predicted = np.clip(model.predict([test_low_image[i].reshape(1,SIZE, SIZE,3),
                         test_low_image[i].reshape(1,SIZE, SIZE,3)]),0.0,1.0).reshape(SIZE, SIZE,3)
    plot_images(test_high_image[i],test_low_image[i],predicted)
    print('PSNR', PSNR(test_high_image[i],predicted))



# # Benchmark images

# In[31]:


def downsample_image(image,scale):
    x=tf.image.resize(image / 255,(image.shape[0]//scale, image.shape[1]//scale))
    x=tf.image.resize(x,(image.shape[0], image.shape[1]), method = tf.image.ResizeMethod.BICUBIC)
    return x


def sorted_alphanumeric(data):  
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)',key)]
    return sorted(data,key = alphanum_key)

# defining the size of the image
SIZE = 128
#high data
high_img = []
path = 'LOL/SR/val/HR/'
files = os.listdir(path)
files = sorted_alphanumeric(files)
for i in tqdm(files):     
        img = cv2.imread(path+i,1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (SIZE, SIZE))
        img = img.astype('float32') / 255.0
        high_img.append(img_to_array(img))

#low daat
low_img = []
path = 'LOL/SR/val/LR/'
files = os.listdir(path)
files = sorted_alphanumeric(files)
for i in tqdm(files):
        img = cv2.imread(path+i,1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (SIZE, SIZE))
        img = img.astype('float32') / 255.0
        low_img.append(img_to_array(img))

#Slicing        
sett_high_image = high_img[1:]
sett_low_image = low_img[1:]
sett_high_image= np.reshape(sett_high_image,(len(sett_high_image),SIZE, SIZE,3))
sett_low_image = np.reshape(sett_low_image,(len(sett_low_image),SIZE,SIZE,3))



# In[32]:


#Show results

def plot_images(high,low,predicted):
    plt.figure(figsize=(15,15))
    plt.subplot(1,3,1)
    plt.title('High Image', color = 'green', fontsize = 20)
    plt.axis('off')
    plt.imshow(high)
    plt.subplot(1,3,2)
    plt.title('Low Image ', color = 'black', fontsize = 20)
    plt.axis('off')
    plt.imshow(low)
    plt.subplot(1,3,3)
    plt.title('Predicted Image ', color = 'Red', fontsize = 20)
    plt.imshow(predicted)   
    plt.axis('off')
    plt.tight_layout()
    plt.show()

for i in range(0,8):    
    predicted = np.clip(model.predict([sett_low_image[i].reshape(1,SIZE, SIZE,3),\
                                       sett_low_image[i].reshape(1,SIZE, SIZE,3)]),0.0,1.0).reshape(SIZE, SIZE,3)
    plot_images(sett_high_image[i],sett_low_image[i],predicted)
    print('PSNR',PSNR(sett_high_image[i],predicted),'dB',"SSIM",tf.image.ssim(sett_high_image[i],predicted,max_val=1))



# In[ ]:




