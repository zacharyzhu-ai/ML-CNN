#!/usr/bin/env python
# coding: utf-8

# # Part 2 - Rock, Paper, Scissors
# 
# 
# This is the 2nd-part of 3-parts series for Convolutional Neural Networks. 
# 1. Basics: binary classification and "Convolution" visualized as transformations through DNN-layers
# 2. Step up: multi-class classification with Rock, Paper, Scissors (this part)
# 3. Production Scale: Horse or Human with Image Augmentation, through convolutions, and Transfer Learning
# 
# Specifically in this part, we will 
# 1. Work with a set of Rock, Paper, Scissors image file from https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps.zip
# 2. Overfitting reduction through image Augmentations such as rotation, width/height shifting, shearing and others
# 3. Build the model
# 4. Train and evaluate accuracy/loss 
# 5. Prediction!
# 
# 
# # 1. Dataset

# In[1]:


get_ipython().system('wget --no-check-certificate     https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps.zip     -O rps.zip')
  
get_ipython().system('wget --no-check-certificate     https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps-test-set.zip     -O rps-test-set.zip')


# In[3]:


import os
import zipfile

local_zip = 'rps.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('pics/')
zip_ref.close()

local_zip = 'rps-test-set.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('pics/')
zip_ref.close()


# Training data

# In[4]:


rock_dir = os.path.join('pics/rps/rock')
paper_dir = os.path.join('pics/rps/paper')
scissors_dir = os.path.join('pics/rps/scissors')

print('total training rock images:', len(os.listdir(rock_dir)))
print('total training paper images:', len(os.listdir(paper_dir)))
print('total training scissors images:', len(os.listdir(scissors_dir)))

rock_files = os.listdir(rock_dir)
paper_files = os.listdir(paper_dir)
scissors_files = os.listdir(scissors_dir)


# Display sample images

# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

pic_index = 2

next_rock = [os.path.join(rock_dir, fname) 
                for fname in rock_files[pic_index-2:pic_index]]
next_paper = [os.path.join(paper_dir, fname) 
                for fname in paper_files[pic_index-2:pic_index]]
next_scissors = [os.path.join(scissors_dir, fname) 
                for fname in scissors_files[pic_index-2:pic_index]]

for i, img_path in enumerate(next_rock+next_paper+next_scissors):
  print(img_path)
  img = mpimg.imread(img_path)
  plt.imshow(img)
  plt.axis('Off')
  plt.show()


# # 2. Augment images
# 
# In addition to rescaling, `keras.preprocessing.image.ImageDataGenerator` also allows for image rotations, width/height shifting, shearing and others.

# In[6]:


import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

TRAINING_DIR = "pics/rps/"
training_datagen = ImageDataGenerator(
      rescale = 1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

VALIDATION_DIR = "pics/rps-test-set/"
validation_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = training_datagen.flow_from_directory(
	TRAINING_DIR,
	target_size=(150,150),
	class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
	VALIDATION_DIR,
	target_size=(150,150),
	class_mode='categorical'
)


# # 3. Model
# 
# - Use 150x150 for input image shape, therefore 
# - Layers
#   1. First convolutional layer is 150x150 size + 3 for RGB color 
#   2. Followed by several convolutional layers with MaxPooling 
#   3. Flatten
#   4. Drop-out
#   5. Dense layer
#   6. `Softmax` and `categorical_crossentropy` for multiclass classification 

# In[8]:


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# # 4.  Training

# In[9]:


epochs = 25 # original 25
history = model.fit(train_generator, epochs=epochs, validation_data = validation_generator)


# # 5. Plot accuracy/loss

# In[11]:


import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()


plt.show()


# # 6. Prediction time

# In[7]:


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


# In[10]:


import numpy as np
from keras.preprocessing import image

uploaded = upload_btn.value

print("estimate in ")
for path in uploaded.keys(): 
  print(path)
    
  img = image.load_img(path, target_size=(150, 150))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)

  images = np.vstack([x])
  classes = model.predict(images, batch_size=10)
  print(classes)


# In[ ]:




