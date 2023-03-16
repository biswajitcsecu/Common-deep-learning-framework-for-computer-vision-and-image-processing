#!/usr/bin/env python
# coding: utf-8

# # Multiclass Semantic Segmentation using Unet

# In[43]:


import os
import keras
import numpy as np
from tqdm import tqdm
import cv2 as cv
from glob import glob
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
import matplotlib.pyplot as plt
from keras import Sequential
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, InputLayer, Layer, Input, Dropout, MaxPool2D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import warnings
warnings.filterwarnings('ignore')


# # Data Load

# In[45]:


#Data processing----------
train_path = 'Insects/train/images/'
image_paths = sorted(glob(train_path + "*.jpg"))
mask_path = 'Insects/train/masks/'
mask_paths = sorted(glob(mask_path + "*.png"))

SIZE = 128

images = np.zeros(shape=(len(image_paths), SIZE, SIZE, 3))
masks = np.zeros(shape=(len(mask_paths), SIZE, SIZE, 3))

for i in tqdm(range(len(image_paths))):
    path = image_paths[i]
    image = cv.imread(path)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = img_to_array(image).astype('float')
    image = image/255.0
    image = tf.image.resize(image,(SIZE, SIZE))
    images[i] = image   
    
for i in tqdm(range(len(mask_paths))):
    path = mask_paths[i]
    mask = cv.imread(path)
    mask = cv.cvtColor(mask, cv.COLOR_BGR2RGB)# for rgb mask  othewse cv.COLOR_BGR2GRY
    mask = img_to_array(mask).astype('float')
    mask = mask/255.0
    mask = tf.image.resize(mask,(SIZE, SIZE))
    masks[i] = mask
    


# # Display samples

# In[46]:


#Samples display

plt.figure(figsize=(8,25))
for i in range(1,21):
    plt.subplot(10,2,i)
    if i%2!=0:
        id = np.random.randint(len(images))
        image = images[id]
        plt.imshow(image)
        plt.axis('off')
    elif i%2==0:
        mask = masks[id]
        plt.imshow(mask)
        plt.axis('off')  
plt.tight_layout()
plt.show()



# # Train data sets

# In[47]:


X_train, y_train = images[:1900], masks[:1900]
X_valid, y_valid = images[1900:], masks[1900:]


# # UNet Autoencoder

# In[48]:


# Encoder Layers

class EncoderLayerBlock(Layer):
    def __init__(self, filters, rate, pooling=True):
        super(EncoderLayerBlock, self).__init__()
        self.filters = filters
        self.rate = rate
        self.pooling = pooling
        self.c1 = Conv2D(self.filters, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')
        self.drop = Dropout(self.rate)
        self.c2 = Conv2D(self.filters, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')
        self.pool = MaxPool2D(pool_size=(2,2))        
        
    def call(self, X):
        x = self.c1(X)
        x = self.drop(x)
        x = self.c2(x)
        
        if self.pooling:
            y = self.pool(x)
            return y, x
        else:
            return x

    def get_config(self):
        base_estimator = super().get_config()
        return { **base_estimator, "filters":self.filters, "rate":self.rate, "pooling":self.pooling }


# In[49]:


#  Decoder Layers

class DecoderLayerBlock(Layer):
    def __init__(self, filters, rate, padding='same'):
        super(DecoderLayerBlock, self).__init__()
        self.filters = filters
        self.rate = rate
        self.cT = Conv2DTranspose(self.filters, kernel_size=3, strides=2, padding=padding)
        self.next = EncoderLayerBlock(self.filters, self.rate, pooling=False)
        
    def call(self, X):
        X, skip_X = X
        x = self.cT(X)
        c1 = concatenate([x, skip_X])
        y = self.next(c1)
        return y 
    
    def get_config(self):
        base_estimator = super().get_config()
        return {**base_estimator,"filters":self.filters,"rate":self.rate, }


# In[50]:


#  Callback 

class ShowProgress(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        id = np.random.randint(len(X_valid))
        rand_img = X_valid[id][np.newaxis,...]
        pred_mask = self.model.predict(rand_img)[0]
        true_mask = y_valid[id]
        
        plt.subplot(1,3,1)
        plt.imshow(rand_img[0])
        plt.title("Original Image")
        plt.axis('off')
        
        plt.subplot(1,3,2)
        plt.imshow(pred_mask)
        plt.title("Predicted Mask")
        plt.axis('off')
        
        plt.subplot(1,3,3)
        plt.imshow(true_mask)
        plt.title("True Mask")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()


# # Train model

# In[51]:


#Train model
with tf.device('cpu'):
    # Input Layer 
    input_layer = Input(shape=X_train.shape[-3:])

    # Encoder
    p1, c1 = EncoderLayerBlock(16,0.1)(input_layer)
    p2, c2 = EncoderLayerBlock(32,0.1)(p1)
    p3, c3 = EncoderLayerBlock(64,0.2)(p2)
    p4, c4 = EncoderLayerBlock(128,0.2)(p3)

    # Encoding Layer
    c5 = EncoderLayerBlock(256,0.3,pooling=False)(p4)

    # Decoder
    d1 = DecoderLayerBlock(128,0.2)([c5, c4])
    d2 = DecoderLayerBlock(64,0.2)([d1, c3])
    d3 = DecoderLayerBlock(32,0.2)([d2, c2])
    d4 = DecoderLayerBlock(16,0.2)([d3, c1])

    # Output layer
    output = Conv2D(3,kernel_size=1,strides=1,padding='same',activation='sigmoid')(d4)
    
    # U-Net Model
    model = keras.models.Model(inputs=[input_layer],outputs=[output], )

    # Compiling
    model.compile( 
        loss='binary_crossentropy',optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy', keras.metrics.MeanIoU(num_classes=2)]
    )

    # Callbacks 
    callbacks =[
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint("UNet-Colorizer.h5", save_best_only=True),
        ShowProgress()
    ]

    # Train The Model
    nepochs=2  
    history = model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        epochs=nepochs,
        callbacks=callbacks,
        verbose=1,
        use_multiprocessing=True
     )
    


# # Evaluation

# In[58]:


#Data loading for validation
train_path = 'Insects/val/images/'
image_paths = sorted(glob(train_path + "*.jpg"))
mask_path = 'Insects/val/masks/'
mask_paths = sorted(glob(mask_path + "*.png"))

SIZE = 128

images = np.zeros(shape=(len(image_paths), SIZE, SIZE, 3))
masks = np.zeros(shape=(len(mask_paths), SIZE, SIZE, 3))

for i in tqdm(range(len(image_paths))):
    path = image_paths[i]
    image = cv.imread(path)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = img_to_array(image).astype('float')
    image = image/255.0
    image = tf.image.resize(image,(SIZE, SIZE))
    images[i] = image   
    
for i in tqdm(range(len(mask_paths))):
    path = mask_paths[i]
    mask = cv.imread(path)
    mask = cv.cvtColor(mask, cv.COLOR_BGR2RGB)
    mask = img_to_array(mask).astype('float')
    mask = mask/255.0
    mask = tf.image.resize(mask,(SIZE, SIZE))
    masks[i] = mask
    


# In[59]:


def show_image(image,title=None):
    plt.imshow(image)
    if title is not None: plt.title(title)
    plt.axis('off')


# In[60]:


#Display results
plt.figure(figsize=(10,30))
n=0

for i in range(1,31):
    plt.subplot(10,3,i)
    if n==0:
        id = np.random.randint(len(images))
        real_img = images[id][np.newaxis,...]
        pred_mask = model.predict(real_img)[0]
        mask = masks[id]
        show_image(real_img[0], title="Real Image")
        n+=1
    elif n==1:
        show_image(pred_mask, title="Predicted Mask")
        n+=1
    elif n==2:
        show_image(mask, title="Original Mask")
        n=0

plt.tight_layout()        
plt.show()


# In[ ]:




