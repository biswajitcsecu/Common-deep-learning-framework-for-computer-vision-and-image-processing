#!/usr/bin/env python
# coding: utf-8

# In[26]:


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
get_ipython().run_line_magic('matplotlib', 'inline')
import splitfolders
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator


import warnings
warnings.filterwarnings('ignore')


# In[20]:


BATCH = 16
W=IMG_WIDTH = 128
H=IMG_HEIGHT = 128
CH=3


# In[16]:


train_loader = tf.keras.preprocessing.image_dataset_from_directory ('Plant/Train', 
            seed = 123, image_size = (IMG_HEIGHT, IMG_WIDTH), batch_size = BATCH)
test_loader = tf.keras.preprocessing.image_dataset_from_directory ('Plant/Test', 
             seed = 123, image_size = (IMG_HEIGHT, IMG_WIDTH), batch_size = BATCH)
validation_loader = tf.keras.preprocessing.image_dataset_from_directory ('Plant/Val', 
             seed = 123, image_size = (IMG_HEIGHT, IMG_WIDTH), batch_size = BATCH)


# In[17]:


class_names = train_loader.class_names
print (class_names)


# In[19]:


#plot sampels

plt.figure (figsize = (10, 10))
for images, labels in train_loader.take(1):
    for i in range (9):
        ax = plt.subplot (3, 3, i+1)
        plt.imshow (images[i].numpy().astype ("uint8"))
        plt.title (class_names[labels[i]])
        plt.axis("off")



# In[13]:


AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_loader.cache().shuffle(1000).prefetch (buffer_size = AUTOTUNE)
test_dataset = test_loader.cache().prefetch (buffer_size = AUTOTUNE)
val_dataset = validation_loader.cache().prefetch (buffer_size = AUTOTUNE)


# In[21]:


vgg_conv = tf.keras.applications.vgg16.VGG16(
    include_top=False,
    weights='imagenet',
    input_shape=(H, W, CH),
    classes = 3
)



# In[27]:


for layer in vgg_conv.layers[:-8]:
    layer.trainable = False


# In[28]:


x = vgg_conv.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = Dense(3, activation="softmax")(x)
model = tf.keras.Model(vgg_conv.input, x)
model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])


# In[29]:


model.summary()


# In[30]:


nepochs = 10
history = model.fit(
    train_dataset,
    validation_data = val_dataset,
    epochs = nepochs, verbose=1,
)



# In[31]:


loss, accuracy = model.evaluate(test_dataset)


# In[33]:


#performance plots

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(nepochs)

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()



# In[34]:


import scipy as sp

def plot_activation(img):
    pred = model.predict(img[np.newaxis,:,:,:])
    pred_class = np.argmax(pred)
    
    weights = model.layers[-1].get_weights()[0]
    
    class_weights = weights[:, pred_class]
    intermediate = tf.keras.Model(model.input,model.get_layer("block5_conv3").output)
    
    conv_output = intermediate.predict(img[np.newaxis,:,:,:])
    conv_output = np.squeeze(conv_output)
    
    h = int(img.shape[0]/conv_output.shape[0])
    w = int(img.shape[1]/conv_output.shape[1])
    
    act_maps = sp.ndimage.zoom(conv_output, (h, w, 1), order=1)
    out = np.dot(act_maps.reshape((img.shape[0]*img.shape[1],512)), class_weights).reshape(img.shape[0],img.shape[1])
    
    plt.imshow(img.astype('float32').reshape(img.shape[0],img.shape[1],3))
    plt.imshow(out, cmap='jet', alpha=0.35)
    plt.title('Crack' if pred_class == 1 else 'No Crack')
    plt.show()



# In[35]:


#plot results
plt.figure(figsize=(10, 10))
for images, labels in test_loader.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        predictions = model.predict(tf.expand_dims(images[i], 0))
        score = tf.nn.softmax(predictions[0])
        plt.ylabel("Predicted: "+class_names[np.argmax(score)])
        plt.title("Actual: "+class_names[labels[i]])
        plt.gca().axes.yaxis.set_ticklabels([])        
        plt.gca().axes.xaxis.set_ticklabels([])



# In[ ]:




