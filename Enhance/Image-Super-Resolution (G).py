#!/usr/bin/env python
# coding: utf-8

# # Import Libraires

# In[53]:


import os
import re 
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage, misc 
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import img_to_array
from skimage.transform import resize, rescale
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, UpSampling2D, add
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
import tensorflow as tf


# # Data loading

# In[38]:


#Data loading
base_dir = "LOL/"
H,W=128,128
CH=3
imgsz=(H,W)

def load_data(path):
    high_res_images = []
    low_res_images = []
    for dirname, _, filenames in os.walk(path+'low'):
        for filename in filenames:
            img = cv2.imread(os.path.join(dirname, filename))
            img = process_image(img)
            img=cv2.resize(img,imgsz)
            low_res_images.append(img)
        
    for dirname, _, filenames in os.walk(path+'high'):
        for filename in filenames:
            img = cv2.imread(os.path.join(dirname, filename))
            img = process_image(img)
            img=cv2.resize(img,imgsz)
            high_res_images.append(img)
    
    return np.array(low_res_images), np.array(high_res_images)

def process_image(image):
    return image/255     


# # Slicing and Reshaping Images

# In[39]:


#Train and valid data sets
train_x, train_y =  load_data(base_dir+'train/')
val_x, val_y = load_data(base_dir+'test/')

print(train_x.shape )    
print('<----------done----------->')
print(val_x.shape)  


# # Data Visualization

# In[40]:


#Display samples
for i in range(4):
    a = np.random.randint(0,50)
    plt.figure(figsize=(8,8))    
    plt.subplot(1,2,1)    
    plt.imshow(train_x[a],aspect='equal')
    plt.axis('off')
    plt.title("Low-resolution image ", color = 'green', fontsize = 20)
    plt.subplot(1,2,2)
    plt.imshow(train_y[a],aspect='equal')
    plt.axis('off')
    plt.title("High-resolution image", color = 'black', fontsize = 20)
    plt.tight_layout()
plt.show()


# # Defining Model

# In[41]:


#Define model
def CNN():
    input_img = Input(shape=(H, W, CH))
    l1 = Conv2D(64, (3, 3), padding='same', activation='relu',activity_regularizer=regularizers.l1(10e-10))(input_img)
    l2 = Conv2D(64, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l1)

    l3 = MaxPooling2D(padding='same')(l2)
    #l3 = Dropout(0.3)(l3)
    l4 = Conv2D(128, (3, 3),  padding='same', activation='relu',activity_regularizer=regularizers.l1(10e-10))(l3)
    l5 = Conv2D(128, (3, 3), padding='same', activation='relu',activity_regularizer=regularizers.l1(10e-10))(l4)

    l6 = MaxPooling2D(padding='same')(l5)
    l7 = Conv2D(256, (3, 3), padding='same', activation='relu',activity_regularizer=regularizers.l1(10e-10))(l6)
    l7 = Conv2D(256, (3, 3), padding='same', activation='relu',activity_regularizer=regularizers.l1(10e-10))(l7)
    
    l8 = UpSampling2D()(l7)

    l9 = Conv2D(128, (3, 3), padding='same', activation='relu',activity_regularizer=regularizers.l1(10e-10))(l8)
    l10 = Conv2D(128, (3, 3), padding='same', activation='relu',activity_regularizer=regularizers.l1(10e-10))(l9)

    l11 = add([l5, l10])
    l12 = UpSampling2D()(l11)
    l13 = Conv2D(64, (3, 3), padding='same', activation='relu',activity_regularizer=regularizers.l1(10e-10))(l12)
    l14 = Conv2D(64, (3, 3), padding='same', activation='relu',activity_regularizer=regularizers.l1(10e-10))(l13)

    l15 = add([l14, l2])

    decoded = Conv2D(3, (3, 3), padding='same', activation='relu',activity_regularizer=regularizers.l1(10e-10))(l15)

    model = Model(input_img, decoded)
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

#define model
model = CNN()
model.summary()
     


# # Compile

# In[42]:


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4),loss=['mse'], metrics = ['acc'])


# # Fitting model

# In[43]:


#Train model
nepochs = 2
nbatch_size = 64

history=model.fit(train_x, train_y,
          epochs = nepochs,
          batch_size = nbatch_size,
          verbose = 1,
          shuffle = True,
          use_multiprocessing=True
         )


# # Prediction Visualization

# In[45]:


predict_y = model.predict(val_x)

for j, i in enumerate(val_x[:3]):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 8))
    
    ax1.imshow(i,aspect='equal')
    ax1.title.set_text("Low-resolution image")
    ax1.axis('off')
    ax2.imshow(predict_y[j],aspect='equal')
    ax2.title.set_text("Predicted resolution")
    ax2.axis('off')
    ax3.imshow(val_y[j],aspect='equal')
    ax3.title.set_text("High-resolution")
    ax3.axis('off')
    plt.tight_layout()
    
plt.show() 


# In[48]:


#Show predicted results
def plot_images(high,low,predicted):
    plt.figure(figsize=(15,15))
    plt.subplot(1,3,1)
    plt.title('High Image', color = 'green', fontsize = 20)
    plt.imshow(high,aspect='equal')
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.title('Low Image ', color = 'black', fontsize = 20)
    plt.imshow(low,aspect='equal')
    plt.axis('off')
    plt.subplot(1,3,3)
    plt.title('Predicted Image ', color = 'Red', fontsize = 20)
    plt.imshow(predicted,aspect='equal') 
    plt.axis('off')
    plt.show()
    plt.tight_layout()

for i in range(1,10):    
    predicted = np.clip(model.predict(val_x[i].reshape(1,H, W,CH)),0.0,1.0).reshape(H, W,CH)
    plot_images(val_y[i],train_x[i],predicted)


# In[67]:


def downsample_image(image,scale):
    x=tf.image.resize(image / 255,(image.shape[0]//scale, image.shape[1]//scale))
    x=tf.image.resize(x,(image.shape[0], image.shape[1]), method = tf.image.ResizeMethod.BICUBIC)
    return x


def sorted_alphanumeric(data):  
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)',key)]
    return sorted(data,key = alphanum_key)

# defining the size of the image
H = 128
W=128
CH=3

high_img = []
path = 'LOL/SR/val/HR'
files = os.listdir(path)
files = sorted_alphanumeric(files)

for i in tqdm(files):    
        img = cv2.imread(path + '/'+i,1)
        # open cv reads images in BGR format so we have to convert it to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #resizing image
        img = cv2.resize(img, (H, W))
        img = img.astype('float32') / 255.0
        high_img.append(img_to_array(img))


low_img = []
path =  'LOL/SR/val/LR'
files = os.listdir(path)
files = sorted_alphanumeric(files)

for i in tqdm(files):    
        img = cv2.imread(path + '/'+i,1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = downsample_image(img,4)
        #resizing image
        img = cv2.resize(img, (H, W))
        img = img.astype('float32') / 255.0
        #img = downsample_image(img,2)
        low_img.append(img_to_array(img))




# In[68]:


#image spliting and reshape     
sett_high_image = high_img[1:]
sett_low_image = low_img[1:]
sett_high_image= np.reshape(sett_high_image,(len(sett_high_image),H, W,CH))
sett_low_image = np.reshape(sett_low_image,(len(sett_low_image),H,W,CH))

print('<=---------Shape-----------=>')
print(sett_high_image.shape)
print('\n')
print(sett_low_image.shape)
print('<=---------done-----------=>')


# In[69]:


def plot_images(high,low,predicted):
    plt.figure(figsize=(15,15))
    plt.subplot(1,3,1)
    plt.title('High Image', color = 'green', fontsize = 20)
    plt.imshow(high)
    plt.subplot(1,3,2)
    plt.title('Low Image ', color = 'black', fontsize = 20)
    plt.imshow(low)
    plt.subplot(1,3,3)
    plt.title('Predicted Image ', color = 'Red', fontsize = 20)
    plt.imshow(predicted)  
    plt.show()



# In[70]:


def PSNR(y_true,y_pred):
    mse=tf.reduce_mean( (y_true - y_pred) ** 2 )
    return 20 * log10(1 / (mse ** 0.5))

def log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def pixel_MSE(y_true,y_pred):
    return tf.reduce_mean( (y_true - y_pred) ** 2 )


# In[72]:


for i in range(0,8):    
    predicted = np.clip(model.predict(sett_low_image[i].reshape(1,H, W,CH)),0.0,1.0).reshape(H, W,CH)
    plot_images(sett_high_image[i],sett_low_image[i], predicted)
    print('PSNR',PSNR(sett_high_image[i],predicted),'dB',"SSIM",tf.image.ssim(sett_high_image[i],predicted,max_val=1))


# In[ ]:




