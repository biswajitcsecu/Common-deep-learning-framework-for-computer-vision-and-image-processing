#!/usr/bin/env python
# coding: utf-8

# In[31]:


import numpy as np

from skimage import img_as_float
from skimage.io import imread_collection, imread, imsave
from skimage.transform import resize
from skimage.metrics import structural_similarity
from skimage.color import rgb2hsv, hsv2rgb
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPool2D, Conv2DTranspose, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.activations import relu
from tensorflow.keras.metrics import mean_squared_error
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, UpSampling2D, add
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
import tensorflow as tf


# In[32]:


os.environ["CUDA_VISIBLE_DEVICES"] = '0'
physical_devices = list(tf.config.experimental.list_physical_devices('GPU'))
for gpu in physical_devices:
    print(gpu)
    tf.config.experimental.set_memory_growth(gpu,True)
list(tf.config.experimental.list_physical_devices())


# In[36]:


base_dir = "LOL/"
H=W=128
CH=3
def load_data(path):
    high_res_images = []
    low_res_images = []
    for dirname, _, filenames in os.walk(path+'low'):
        for filename in filenames:
            img = cv2.imread(os.path.join(dirname, filename))
            img= cv2.resize(img,(H,W))
            img = process_image(img)
            low_res_images.append(img)
        
    for dirname, _, filenames in os.walk(path+'high'):
        for filename in filenames:
            img = cv2.imread(os.path.join(dirname, filename))
            img= cv2.resize(img,(H,W))
            img = process_image(img)
            high_res_images.append(img)
    
    return np.array(low_res_images), np.array(high_res_images)



def process_image(image):
    return image/255



# In[37]:


train_x, train_y =  load_data(base_dir+'train/')
val_x, val_y = load_data(base_dir+'val/')
train_x.shape


# In[38]:


fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Image Comparison')
ax1.imshow(train_x[44])
ax1.title.set_text("low-res image ")
ax2.imshow(train_y[44])
ax2.title.set_text("high-res image ")


# In[39]:


def build_model():
    input_img = Input(shape=(H, W, CH))
    l1 = Conv2D(64, (3, 3), padding='same', activation='relu', 
                activity_regularizer=regularizers.l1(10e-10))(input_img)
    l2 = Conv2D(64, (3, 3), padding='same', activation='relu', 
                activity_regularizer=regularizers.l1(10e-10))(l1)

    l3 = MaxPooling2D(padding='same')(l2)
    #l3 = Dropout(0.3)(l3)
    l4 = Conv2D(128, (3, 3),  padding='same', activation='relu', 
                activity_regularizer=regularizers.l1(10e-10))(l3)
    l5 = Conv2D(128, (3, 3), padding='same', activation='relu', 
                activity_regularizer=regularizers.l1(10e-10))(l4)

    l6 = MaxPooling2D(padding='same')(l5)
    l7 = Conv2D(256, (3, 3), padding='same', activation='relu', 
                activity_regularizer=regularizers.l1(10e-10))(l6)
    l7 = Conv2D(256, (3, 3), padding='same', activation='relu', 
                activity_regularizer=regularizers.l1(10e-10))(l7)
    
    l8 = UpSampling2D()(l7)

    l9 = Conv2D(128, (3, 3), padding='same', activation='relu',
                activity_regularizer=regularizers.l1(10e-10))(l8)
    l10 = Conv2D(128, (3, 3), padding='same', activation='relu',
                 activity_regularizer=regularizers.l1(10e-10))(l9)

    l11 = add([l5, l10])
    l12 = UpSampling2D()(l11)
    l13 = Conv2D(64, (3, 3), padding='same', activation='relu',
                 activity_regularizer=regularizers.l1(10e-10))(l12)
    l14 = Conv2D(64, (3, 3), padding='same', activation='relu',
                 activity_regularizer=regularizers.l1(10e-10))(l13)

    l15 = add([l14, l2])

    decoded = Conv2D(3, (3, 3), padding='same', activation='relu', 
                     activity_regularizer=regularizers.l1(10e-10))(l15)


    model = Model(input_img, decoded)
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model


# In[40]:


model = build_model()
model.summary()


# In[42]:


nepochs = 5  
nbatch_size = 16
history=model.fit(train_x, train_y, epochs = nepochs,  batch_size = nbatch_size, verbose = 1, shuffle = True)


# In[43]:


predict_y = model.predict(val_x)

for j, i in enumerate(val_x[:3]):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 8))
    
    ax1.imshow(i)
    ax1.title.set_text("low-res image ")
    ax2.imshow(predict_y[j])
    ax2.title.set_text("model's output")
    ax3.imshow(val_y[j])
    ax3.title.set_text("actual-high-res")


# In[ ]:




