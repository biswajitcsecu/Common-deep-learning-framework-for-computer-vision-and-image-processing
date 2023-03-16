#!/usr/bin/env python
# coding: utf-8

# In[56]:


import tensorflow as tf
import cv2
import os
import random
import numpy as np
import glob
import keract
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D 
from tensorflow.keras.layers import UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization
from tensorflow.keras.layers import Dropout, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import warnings

warnings.filterwarnings('ignore')
random.seed(23)


# In[57]:


## resizing images 
def img_resize(image, y_dim, x_dim):
    resized_img = cv2.resize(image, (y_dim,x_dim))
    return resized_img


# In[58]:


image_path = "ForestAerial/images/*.jpg"
mask_path = "ForestAerial/masks/*.jpg"
image_names = sorted(glob.glob(image_path), key=lambda x: x.split('.')[0])
mask_names = sorted(glob.glob(mask_path), key=lambda x: x.split('.')[0])


# In[59]:


H=128
W=128
CH=3

image_array = []
mask_array = []

#images
for image in image_names:
    img = cv2.imread(image, -1)
    img = img_resize(img, H, W)
    image_array.append(img)    
image_array = np.array(image_array)


#masks
for mask in mask_names:
    msk = cv2.imread(mask, 0)
    msk = img_resize(msk, H, W)
    mask_array.append(msk)    
mask_array = np.array(mask_array)


# In[60]:


num = random.randint(0,100)

def normalize_array(arr):
    return arr/255.0

X = normalize_array(image_array)
y = normalize_array(mask_array)


# In[61]:


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


# In[62]:


## splitting the image into train and test 
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1, random_state=23)


# In[63]:


## creating a unet model

def conv_block(inputs=None, n_filters=32, dropout_prob=0, max_pooling=True):
    
    conv = Conv2D(n_filters, 3,  activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv = Conv2D(n_filters, 3,activation='relu',padding='same', kernel_initializer='he_normal')(conv)
    
    if dropout_prob > 0:
        conv = Dropout(dropout_prob)(conv)
         
    if max_pooling:
        next_layer = MaxPooling2D(pool_size=(2,2))(conv)
        
    else:
        next_layer = conv
        
    skip_connection = conv
    
    return next_layer, skip_connection


# In[64]:


def upsampling(expansive_input, contractive_input, n_filters=32):
    up = Conv2DTranspose(n_filters,3,strides=(2,2),padding='same')(expansive_input)    
    merge = concatenate([up, contractive_input], axis=3)
    
    conv = Conv2D(n_filters,3,activation='relu',padding='same',kernel_initializer='he_normal')(merge)
    conv = Conv2D(n_filters,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv)

    return conv


# In[65]:


def unet(input_size=(H, W, CH), n_filters=32):

    inputs = Input(input_size)

    cblock1 = conv_block(inputs, n_filters)
    cblock2 = conv_block(cblock1[0], 2*n_filters)
    cblock3 = conv_block(cblock2[0], 2**2*n_filters)
    cblock4 = conv_block(cblock3[0], 2**3*n_filters, dropout_prob=0.3) 
    cblock5 = conv_block(cblock4[0], 2**4*n_filters, dropout_prob=0.3, max_pooling=False) 

    ublock6 = upsampling(cblock5[0], cblock4[1],  2**3*n_filters)
    ublock7 = upsampling(ublock6, cblock3[1],  2**2*n_filters)
    ublock8 = upsampling(ublock7, cblock2[1],  2*n_filters)
    ublock9 = upsampling(ublock8, cblock1[1],  n_filters)

    conv9 = Conv2D(n_filters,3,activation='relu',padding='same',kernel_initializer='he_normal')(ublock9)
    conv10 = Conv2D(1, 1, padding='same',activation='sigmoid')(conv9)
    
    model = tf.keras.Model(inputs=inputs, outputs=conv10)

    return model


# In[66]:


## creating and compiling a model

model = unet(n_filters=32,input_size=(H,W,CH))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[21]:


#model summary
model.summary()


# In[22]:


early_stopping_cb = EarlyStopping(patience=25,restore_best_weights=True)
model_checkpoint_cb = ModelCheckpoint("forest_image_segmentor_model.h5",save_best_only=True)


# In[25]:


num_epochs = 2

history = model.fit(X_train,y_train,epochs=num_epochs,callbacks=[early_stopping_cb], 
                    validation_data=(X_test, y_test), verbose=1, use_multiprocessing=True)


# In[31]:


## creating an accuracy graph for training and testing data
plt.plot(history.history['accuracy'],color='yellow',label='training accuracy')
plt.plot(history.history['val_accuracy'],color='red',label='Testing accuracy')
plt.legend()
plt.show()


# In[32]:


# creating an loss graph for training and testing data
plt.plot(history.history['loss'],color='yellow',label='training loss')
plt.plot(history.history['val_loss'],color='red',label='Testing loss')
plt.legend()
plt.show()



# In[33]:


figure, axes = plt.subplots(5,3, figsize=(25,25))

for i in range(0,5):
    rand_num = random.randint(0,400)
    original_img = X_test[rand_num]
    axes[i,0].imshow(original_img)
    axes[i,0].title.set_text('Original Image')
    
    original_mask = y_test[rand_num]
    axes[i,1].imshow(original_mask)
    axes[i,1].title.set_text('Original Mask')
    
    original_img = np.expand_dims(original_img, axis=0)
    predicted_mask = model.predict(original_img).reshape(H,W)
    axes[i,2].imshow(predicted_mask)
    axes[i,2].title.set_text('Predicted Mask')


# In[49]:


image = load_img('ForestAerial/images/111335_sat_08.jpg', target_size= (H, W))
image = img_to_array(image)
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
image = preprocess_input(image)
y_hat = model.predict(image)


# In[54]:


layers=['conv2d', 'conv2d_4', 'conv2d_8', 'conv2d_10', 
        'conv2d_22', 'conv2d_28', 'conv2d_34', 'conv2d_40',
        'conv2d_52', 'conv2d_61', 'conv2d_67', 'conv2d_70']


# In[55]:


activations= keract.get_activations(model, image, layer_names= layers, 
                                    nodes_to_evaluate= None, output_format= 'simple', auto_compile= True)
keract.display_activations(activations, cmap='viridis', save= False, directory= 'activations')



# In[ ]:




