#!/usr/bin/env python
# coding: utf-8

# In[105]:


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
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
from tensorflow import keras

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')


# In[78]:


BATCH = 32
W=IMG_WIDTH = 128
H=IMG_HEIGHT = 128
CH=3


# In[79]:


train_loader = tf.keras.preprocessing.image_dataset_from_directory ('Seedlings/train', 
            seed = 123, image_size = (IMG_HEIGHT, IMG_WIDTH), batch_size = BATCH)
validation_loader = tf.keras.preprocessing.image_dataset_from_directory ('Seedlings/val', 
             seed = 123, image_size = (IMG_HEIGHT, IMG_WIDTH), batch_size = BATCH)


# In[80]:


class_names = train_loader.class_names
print (class_names)


# In[81]:


plt.figure (figsize = (10, 10))
for images, labels in train_loader.take(1):
    for i in range (12):
        ax = plt.subplot (3, 4, i+1)
        plt.imshow (images[i].numpy().astype ("uint8"))
        plt.title (class_names[labels[i]])
        plt.axis("off")


# In[82]:


AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_loader.cache().shuffle(1000).prefetch (buffer_size = AUTOTUNE)
val_dataset = validation_loader.cache().prefetch (buffer_size = AUTOTUNE)


# In[83]:


nclass=len(class_names)
vgg_conv = tf.keras.applications.vgg16.VGG16(
    include_top=False,
    weights='imagenet',
    input_shape=(H, W, CH),
    classes = nclass
)

for layer in vgg_conv.layers[:-8]:
    layer.trainable = False


# In[90]:


#Model configure
x = vgg_conv.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = Dense(nclass, activation=tf.keras.layers.LeakyReLU(alpha=0.01))(x) #activation="softmax"

model = tf.keras.Model(vgg_conv.input, x)

model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, ignore_class=None),
              optimizer = optimizers.SGD(learning_rate=0.0001, momentum=0.9), metrics=["accuracy"])


# In[91]:


model.summary()


# In[93]:


nepochs = 1
history = model.fit( train_dataset, validation_data = val_dataset, epochs = nepochs, verbose=1, use_multiprocessing=True)


# In[94]:


loss, accuracy = model.evaluate(val_dataset)


# In[95]:


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



# In[96]:


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



# In[97]:


#plot results
plt.figure(figsize=(10, 10))
for images, labels in validation_loader.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        predictions = model.predict(tf.expand_dims(images[i], 0))
        score = tf.nn.softmax(predictions[0])
        plt.ylabel("Predicted: "+class_names[np.argmax(score)])
        plt.title("Actual: "+class_names[labels[i]])
        plt.gca().axes.yaxis.set_ticklabels([])        
        plt.gca().axes.xaxis.set_ticklabels([])


# In[163]:


test_pred = model.predict(val_dataset)


# In[164]:


for image_batch, label_batch in train_dataset.take(1):
    mimage = image_batch[0]
    mlabel = label_batch[0]

plt.imshow(mimage)


# In[ ]:




