#!/usr/bin/env python
# coding: utf-8

# # Part 3 - Production Scale: Horse or Human, with Google InceptionV3 model through Transfer Learning
# 
# 
# This is the final part of 3-parts series for Convolutional Neural Networks. 
# 1. Basics: binary classification and "Convolution" visualized as transformations through DNN-layers
# 2. Step up: multi-class classification with Rock, Paper, Scissors 
# 3. Production Scale: Horse or Human with Image Augmentation, through convolutions, and Transfer Learning (this part)
# 
# Specifically in this part, we will leverage the Version 3 of "Inception" model from Google, which is trained on more than 1million images from ImageNet. 
# 1. Load a set of computer generated Horse/Human images from  https://storage.googleapis.com/laurencemoroney-blog.appspot.com
# 2. Preprocessing and image Augmentations with `keras.preprocessing.image.ImageDataGenerator` to prepare train/validation data
# 3. Build the basic model
# 4. Visualize  image transformations through each iterations of "Convolution"
# 5. Train and evaluate accuracy/loss 
# 6. Apply transferred learning for large-scale, massive neural-networks
# 
# 
# 
# # 1. Dataset

# In[1]:


get_ipython().system('wget --no-check-certificate     https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip     -O horse-or-human.zip')

get_ipython().system('wget --no-check-certificate     https://storage.googleapis.com/laurencemoroney-blog.appspot.com/validation-horse-or-human.zip     -O validation-horse-or-human.zip')


# In[2]:


import os
import zipfile

training_dir = 'horse-or-human/training/'
validation_dir = 'horse-or-human/validation/'


# In[3]:


training_file_name = "horse-or-human.zip"
zip_ref = zipfile.ZipFile(training_file_name, 'r')
zip_ref.extractall(training_dir)
zip_ref.close()

validation_file_name = "validation-horse-or-human.zip"
zip_ref = zipfile.ZipFile(validation_file_name, 'r')
zip_ref.extractall(validation_dir)
zip_ref.close()


# In[4]:


# Directory with our training horse pictures
train_horse_dir = os.path.join(training_dir, 'horses')

# Directory with our training human pictures
train_human_dir = os.path.join(training_dir, 'humans')

# Directory with our training horse pictures
validation_horse_dir = os.path.join(validation_dir, 'horses')

# Directory with our training human pictures
validation_human_dir = os.path.join(validation_dir, 'humans')

train_horse_names = os.listdir(train_horse_dir)
print("10 train_horse_names", train_horse_names[:10])

train_human_names = os.listdir(train_human_dir)
print("10 train_human_names", train_human_names[:10])

validation_horse_hames = os.listdir(validation_horse_dir)
print("10 validation_horse_hames", validation_horse_hames[:10])

validation_human_names = os.listdir(validation_human_dir)
print("10 validation_human_names", validation_human_names[:10])


# # 2. Image Augmentations

# In[5]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        training_dir,  
        target_size=(300, 300),  # 300x300 for basic_model 
        batch_size=128,
        class_mode='binary')

validation_datagen = ImageDataGenerator(rescale=1/255)

validation_generator = train_datagen.flow_from_directory(
        validation_dir,
        target_size=(300, 300),
        class_mode='binary')


# # 3. Basic Model 

# In[6]:


import tensorflow as tf

basic_model = tf.keras.models.Sequential([
    # This is the first convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

basic_model.summary()


# In[7]:


from tensorflow.keras.optimizers import RMSprop

basic_model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(learning_rate=0.001),
              metrics=['acc'])


# # 4. Visualize image transformations through each "Convolutions"
# 
# See how an input gets transformed as it goes through the model 
# 1. Pick a random image from the training set
# 2. Generate a figure where each row is the output of a layer, and each image in the row is a specific filter in that output `feature map`. 
# 3. Rerun this cell to generate intermediate representations for a variety of training images.

# In[8]:


import numpy as np
import random
from tensorflow.keras.preprocessing.image import img_to_array, load_img

get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# Prepare a random input image from the training set.
horse_img_files = [os.path.join(train_horse_dir, f) for f in train_horse_names]
human_img_files = [os.path.join(train_human_dir, f) for f in train_human_names]
img_path = random.choice(horse_img_files + human_img_files)

img = mpimg.imread(img_path)
plt.title(img_path.rsplit('\\',1)[1])
plt.imshow(img)

img = load_img(img_path, target_size=(300, 300))  # this is a PIL image
x = img_to_array(img)  # Numpy array with shape (150, 150, 3)
x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 150, 150, 3)

# Rescale by 1/255
x /= 255


# Define a new Model that will take an image as input
# Output intermediate representations for all layers after that
successive_outputs = [layer.output for layer in basic_model.layers[1:]]
#visualization_model = Model(img_input, successive_outputs)
visualization_model = tf.keras.models.Model(inputs = basic_model.input, outputs = successive_outputs)


# Run the image through the network, thus obtaining all intermediate representations for this image.
successive_feature_maps = visualization_model.predict(x)

# Get names of the layers
layer_names = [layer.name for layer in basic_model.layers]

# Display the representations
for layer_name, feature_map in zip(layer_names, successive_feature_maps):
  if len(feature_map.shape) == 4:
    # Just do this for the conv / maxpool layers, not the fully-connected layers
    n_features = feature_map.shape[-1]  # number of features in feature map
    n_features = 5
    # The feature map has shape (1, size, size, n_features)
    size = feature_map.shape[1]
    # We will tile our images in this matrix
    display_grid = np.zeros((size, size * n_features))
    for i in range(n_features):
      # Postprocess the feature to make it visually palatable
      x = feature_map[0, :, :, i]
      x -= x.mean()
      x /= x.std()
      x *= 64
      x += 128
      x = np.clip(x, 0, 255).astype('uint8')
      # We'll tile each filter into this big horizontal grid
      display_grid[:, i * size : (i + 1) * size] = x
    # Display the grid
    scale = 20. / n_features
    plt.figure(figsize=(scale * n_features, scale))
    #plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')


# # 5. Training the basic model

# In[9]:


basic_history = basic_model.fit(
      train_generator,
      steps_per_epoch=8,  
      epochs=15,
      verbose=1,
      validation_data=validation_generator)


# # 6. Plot accuracy/loss

# In[10]:


import matplotlib.pyplot as plt

def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()


# In[11]:


plot_graphs(basic_history, "acc")
plot_graphs(basic_history, "loss")


# # 7. Transfer Learning 
# 
# The ability to leverage pre-trained models is tremendously advantageous, since one could skip weeks of training time of very deep networks. One could just use the features it has learned, tweak against for your dataset and apply own necessary/dense layers. 
# 
# In our case, we will 
# 1. Get the `InceptionV3` pre-trained model
# 2. Set the input shape: 150x150x3 (feel free to set to larger-size i.e. through GCP/AWS/Azure) 
# 3. Pick and freeze the convolution layers to take advantage of the features it has learned 
# 4. Add dense layers
# 
# Note that
# 1. Fetch the pretrained weights of the InceptionV3 model 
# 2. Remove the fully connected layer at the end since we will replace it
# 3. pecify the input shape that your model will accept. Lastly, you want to freeze the weights of these layers because they have been trained already.

# ### Set input data of 150x150 (feel free to set to larger-size i.e. through GCP/AWS/Azure) 

# In[12]:


# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        training_dir,  
        target_size=(150, 150),  # 300x300 for basic_model 
        batch_size=20,
        class_mode='binary')

# Flow validation images in batches of 20 using test_datagen generator
validation_generator = train_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')


# ### Import `Inception` model

# In[13]:


import urllib.request
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import layers
from tensorflow.keras import Model

from tensorflow.keras.optimizers import RMSprop

weights_url = "https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
weights_file = "inception_v3.h5"
urllib.request.urlretrieve(weights_url, weights_file)

pre_trained_model = InceptionV3(input_shape=(150, 150, 3),
                                include_top=False,
                                weights=None)

pre_trained_model.load_weights(weights_file)

for layer in pre_trained_model.layers:
    layer.trainable = False

pre_trained_model.summary()


# ### Pick the convolution layer

# In[14]:


last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output

# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)
# Add a final sigmoid layer for classification
x = layers.Dense(1, activation='sigmoid')(x)

pretrained_model = Model(pre_trained_model.input, x)

pretrained_model.compile(optimizer=RMSprop(lr=0.0001),
              loss='binary_crossentropy',
              metrics=['acc'])

pretrained_model.summary()


# In[15]:


pretrained_history = pretrained_model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=20)


# In[16]:


plot_graphs(pretrained_history, "acc")
plot_graphs(pretrained_history, "loss")


# # 8. Prediction 
# 

# In[17]:


from ipywidgets import FileUpload

def on_upload_change(change):
#    print("change is", change)
    if not change.new:
        return
    up = change.owner
    for uploadFile,data in up.value.items():
        print(f'getting [{uploadFile}]')
        with open(uploadFile, 'wb') as f:
            f.write(data['content'])
    up.value.clear()
    up._counter = 0

upload_btn = FileUpload()
uploadFile = upload_btn.observe(on_upload_change, names='_counter')
print("uploadFile is", uploadFile)
upload_btn


# In[18]:


import numpy as np
from keras.preprocessing import image

uploaded = upload_btn.value

for path in uploaded.keys(): 
  print("file", path)
    
  img = image.load_img(path, target_size=(150, 150))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)

  images = np.vstack([x])
  classes = pretrained_model.predict(images, batch_size=10)
  print(classes)


# In[ ]:




