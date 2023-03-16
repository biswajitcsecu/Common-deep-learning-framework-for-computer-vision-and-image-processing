#!/usr/bin/env python
# coding: utf-8

# # Multiclass Polyp Semantic Segmentation using DeepLabV3

# In[13]:


import os
import keras
import numpy as np
from tqdm import tqdm
import cv2 as cv
from glob import glob
from keras import *
import tensorflow as tf
import tensorflow.data as tfd
import tensorflow.image as tfi
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from keras.layers import Concatenate
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, InputLayer, Layer, Input, Dropout, MaxPool2D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import UpSampling2D, Concatenate
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.utils import plot_model
from tf_explain.core.grad_cam import GradCAM


import warnings
warnings.filterwarnings('ignore')


# # Data Load

# In[41]:


#Data processing----------
H = 128
W = 128
CH=3
images = np.zeros(shape=(len(image_paths), H, W, CH))
masks = np.zeros(shape=(len(mask_paths), H, W, CH))

#Data processing----------
train_path = 'Polyp/train/images/'
image_paths = sorted(glob(train_path + "*.png"))
mask_path = 'Polyp/train/masks/'
mask_paths = sorted(glob(mask_path + "*.png"))

#pack up
for i in tqdm(range(len(image_paths))):
    path = image_paths[i]
    image = cv.imread(path)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = img_to_array(image).astype('float')
    image = image/255.0
    image = tf.image.resize(image,(H, W))
    images[i] = image   
    
for i in tqdm(range(len(mask_paths))):
    path = mask_paths[i]
    mask = cv.imread(path)
    mask = cv.cvtColor(mask, cv.COLOR_BGR2RGB)# for rgb mask  othewse cv.COLOR_BGR2GRAY
    mask = img_to_array(mask).astype('float')
    mask = mask/255.0
    mask = tf.image.resize(mask,(H, W))
    masks[i] = mask  
    
#info
print('<=-------------------Dim---------------=>')
print('Images size', len(images))
print('\n')
print('Masks size', len(masks))
print('<=-------------------Done---------------=>')


# # Display samples

# In[42]:


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



# # Train data Loading

# In[43]:


# Params
AUTO = tfd.AUTOTUNE
BATCH_SIZE =32

X_train, y_train = images[:500], masks[:500]
X_valid, y_valid = images[500:], masks[500:]

print('#<=--------train-----------------=>#')
# Converting to TF Data
train_ds = tfd.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=True)
train_ds = train_ds.prefetch(AUTO)

print('#---------valid------------------#')
# Converting to TF Data
valid_ds = tfd.Dataset.from_tensor_slices((X_valid, y_valid))
valid_ds = valid_ds.batch(BATCH_SIZE, drop_remainder=True)
valid_ds = valid_ds.prefetch(AUTO)

print('#----------test-----------------#')
# Converting to TF Data
test_ds = tfd.Dataset.from_tensor_slices((X_valid, y_valid))
test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=True)
test_ds = test_ds.prefetch(AUTO)
print('#-----------done----------------#')


# # Data Visualization

# In[44]:


def show_maps(data, n_images=10, model=None, SIZE=(20,10), ALPHA=0.5, explain=False):
    
    # plot Configurations
    if model is not None:
        n_cols = 4
    else:
        n_cols = 3
    
    # Select the Data
    images, label_maps = next(iter(data))
    
    if model is None:
        # Create N plots where N = Number of Images
        for image_no in range(n_images):

            # Figure Size
            plt.figure(figsize=SIZE)

            # Select Image and Label Map 
            id = np.random.randint(len(images))
            image, label_map = images[id], label_maps[id]

            # Plot Image 
            plt.subplot(1, n_cols, 1)
            plt.imshow(image)
            plt.title("Original Image")
            plt.axis('off')

            # Plot Label Map
            plt.subplot(1, n_cols, 2)
            plt.imshow(label_map)
            plt.title('Original Label Map')
            plt.axis('off')

            # Plot Mixed Overlap
            plt.subplot(1, n_cols, 3)
            plt.imshow(image)
            plt.imshow(label_map, alpha=ALPHA)
            plt.title("Overlap")
            plt.axis('off')

            # Final Show
            plt.show()
    elif explain:
        n_cols = 5
        exp = GradCAM()
        # Create N plots where N = Number of Images
        for image_no in range(n_images):

            # Select Image and Label Map
            images, label_maps = valid_images, valid_label_maps
            id = np.random.randint(len(images))
            image, label_map = images[id], label_maps[id]
            pred_map = model.predict(image[np.newaxis, ...])[0]
            
            # Grad Cam
            cam = exp.explain(
                validation_data=(image[np.newaxis,...], label_map),
                class_index=1,
                layer_name='TopCB-2',
                model=model
            )
            
            # Figure Size
            plt.figure(figsize=SIZE)

            # Plot Image 
            plt.subplot(1, n_cols, 1)
            plt.imshow(image)
            plt.title("Original Image")
            plt.axis('off')

            # Plot Original Label Map
            plt.subplot(1, n_cols, 2)
            plt.imshow(label_map)
            plt.title('Original Label Map')
            plt.axis('off')
            
            # Plot Predicted Label Map
            plt.subplot(1, n_cols, 3)
            plt.imshow(pred_map)
            plt.title('Predicted Label Map')
            plt.axis('off')
            
            # Plot Mixed Overlap
            plt.subplot(1, n_cols, 4)
            plt.imshow(image)
            plt.imshow(pred_map, alpha=ALPHA)
            plt.title("Overlap")
            plt.axis('off')
            
            # Plot Grad Cam
            plt.subplot(1, n_cols, 5)
            plt.imshow(cam)
            plt.title("Grad Cam")
            plt.axis('off')

            # Final Show
            plt.show()
        
    else:
        # Create N plots where N = Number of Images
        for image_no in range(n_images):

            # Figure Size
            plt.figure(figsize=SIZE)

            # Select Image and Label Map 
            id = np.random.randint(len(images))
            image, label_map = images[id], label_maps[id]
            pred_map = model.predict(image[np.newaxis, ...])[0]

            # Plot Image 
            plt.subplot(1, n_cols, 1)
            plt.imshow(image)
            plt.title("Original Image")
            plt.axis('off')

            # Plot Original Label Map
            plt.subplot(1, n_cols, 2)
            plt.imshow(label_map)
            plt.title('Original Label Map')
            plt.axis('off')
            
            # Plot Predicted Label Map
            plt.subplot(1, n_cols, 3)
            plt.imshow(pred_map)
            plt.title('Predicted Label Map')
            plt.axis('off')
            
            # Plot Mixed Overlap
            plt.subplot(1, n_cols, 4)
            plt.imshow(image)
            plt.imshow(pred_map, alpha=ALPHA)
            plt.title("Overlap")
            plt.axis('off')

            # Final Show
            plt.show()


# In[45]:


show_maps(data=train_ds)


# # DeepLabV3

# In[46]:


class ConvBlock(Layer):
    
    def __init__(self, filters=256, kernel_size=3, dilation_rate=1, **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        
        self.net = Sequential([
            Conv2D(filters=filters, kernel_size=kernel_size, strides=1, padding='same', 
                   kernel_initializer='he_normal', dilation_rate=dilation_rate, use_bias=True),
            BatchNormalization(),
            ReLU()
        ])
        
    def call(self, X): return self.net(X)
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "filters":self.filters, "kernel_size":self.kernel_size,
                "dilation_rate":self.dilation_rate}


# In[47]:


def AtrousSpatialPyramidPooling(X):
    
    _, height, width, _ = X.shape
    
    y = AveragePooling2D(pool_size=(height, width), name="ASPP-AvgPool")(X)
    y = ConvBlock(kernel_size=1, name="ASPP-ImagePool")(y)
    image_pool = UpSampling2D(size=(height//y.shape[1], width//y.shape[2]),
                              interpolation='bilinear', name="ASPP-UpSample")(y)
        
    conv_1 = ConvBlock(kernel_size=1, dilation_rate=1, name="ASPP-Conv1")(X)
    conv_6 = ConvBlock(kernel_size=3, dilation_rate=6, name="ASPP-Conv6")(X)
    conv_12 = ConvBlock(kernel_size=3, dilation_rate=12, name="ASPP-Conv12")(X)
    conv_18 = ConvBlock(kernel_size=3, dilation_rate=18, name="ASPP-Conv18")(X)
    
    concat = Concatenate(axis=-1, name="ASPP-Concat")([image_pool, conv_1, conv_6, conv_12, conv_18])
    out = ConvBlock(kernel_size=1, name="ASPP-Output")(concat)
    
    return out


# In[48]:


# Input Layer
H = 128
W=128
CH=3

InputL = Input(shape=(H, W, CH), name="Input-Layer")

# Base Model
resnet50 = ResNet50(include_top=False, input_tensor=InputL, weights='imagenet')

# DCNN Output
DCNN = resnet50.get_layer('conv4_block6_2_relu').output
ASPP = AtrousSpatialPyramidPooling(DCNN)
ASPP = UpSampling2D(size=(H//4//ASPP.shape[1], W//4//ASPP.shape[2]),
                    interpolation='bilinear', name="Atrous-Upscale")(ASPP)

# Low Level Features
LLF = resnet50.get_layer('conv2_block3_2_relu').output
LLF = ConvBlock(filters=48, kernel_size=1, name="LLF-ConvBlock")(LLF)

# Combine
combine = Concatenate(axis=-1, name="Combine-Features")([ASPP, LLF])
y = ConvBlock(name="TopCB-1")(combine)
y = ConvBlock(name="TopCB-2")(y)
y = UpSampling2D(size=(H//y.shape[1], W//y.shape[1]), interpolation='bilinear', name="Top-UpSample")(y)
LabelMap = Conv2D(filters=3, kernel_size=1, strides=1, activation='sigmoid', padding='same', name="OutputLayer")(y)

# model 
model = Model(InputL, LabelMap, name="DeepLabV3-Plus")
model.summary()


# # Model Visualization

# In[49]:


plot_model(model, "DeepLabV3+.png", dpi=96, show_shapes=True)


# # Training

# In[50]:


class ShowProgress(Callback):
    def on_epoch_end(self, epochs, logs=None):
        show_maps(data=valid_ds, model=self.model, n_images=1)


# In[51]:


model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    metrics=['accuracy']
)


# In[52]:


# Callbacks 
cbs = [ModelCheckpoint("DeepLabV3+.h5",save_best_only=False),ShowProgress()]


# In[ ]:


nepochs=25
history = model.fit(train_ds, validation_data=valid_ds,
                    epochs=nepochs, callbacks=cbs,
                    verbose=1,use_multiprocessing=True
                   )


# # Model Preedictions 

# In[ ]:


show_maps(data=test_ds, model=model)


# # Model Training 

# In[ ]:


nepochs=25
history = model.fit(train_ds, validation_data=valid_ds, 
                    epochs=nepochs, callbacks=cbs,
                   verbose=1,use_multiprocessing=True
                   )


# # Model Predictions 

# In[ ]:


show_maps(data=test_ds, model=model)


# # Grad-CAM technique - Bonus

# In[ ]:


show_maps(data=test_ds, model=model, explain=True)


# In[ ]:





# In[ ]:




