import wget
import os
import tensorflow as tf
print(tf.__version__)

############## workaround ###########
#config = tf.compat.v1.ConfigProto(gpu_options =
#                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
#                        )
#config.gpu_options.allow_growth = True
#session = tf.compat.v1.Session(config=config)
#tf.compat.v1.keras.backend.set_session(session)


######### download and unzip ##############
import os
import wget

url = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip'
out = f'{os.getcwd()}\\tmp\\horse-or-human.zip'

if not os.path.isfile(out):
    wget.download(url, out=out)
    print('Download1 Finished')
else:
    print('Download1 Skipped')

url2 = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/validation-horse-or-human.zip'
out2 = f'{os.getcwd()}\\tmp\\validation-horse-or-human.zip'

if not os.path.isfile(out2):
    wget.download(url, out=out2)
    print('Download2 Finished')
else:
    print('Download2 Skipped')

import zipfile

local_zip = f'{os.getcwd()}\\tmp\\horse-or-human.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall(f'{os.getcwd()}\\tmp\\horse-or-human')
local_zip = f'{os.getcwd()}\\tmp\\validation-horse-or-human.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall(f'{os.getcwd()}\\tmp\\validation-horse-or-human')
zip_ref.close()

# Directory with our training horse pictures
train_horse_dir = os.path.join(f'{os.getcwd()}\\tmp\\horse-or-human\\horses')

# Directory with our training human pictures
train_human_dir = os.path.join(f'{os.getcwd()}\\tmp\\horse-or-human\\humans')

# Directory with our training horse pictures
validation_horse_dir = os.path.join(f'{os.getcwd()}\\tmp\\validation-horse-or-human\\horses')

# Directory with our training human pictures
validation_human_dir = os.path.join(f'{os.getcwd()}\\tmp\\validation-horse-or-human\\humans')


######### folder contents ##############
train_horse_names = os.listdir(train_horse_dir)
print(train_horse_names[:10])

train_human_names = os.listdir(train_human_dir)
print(train_human_names[:10])

validation_horse_hames = os.listdir(validation_horse_dir)
print(validation_horse_hames[:10])

validation_human_names = os.listdir(validation_human_dir)
print(validation_human_names[:10])

print('total training horse images:', len(os.listdir(train_horse_dir)))
print('total training human images:', len(os.listdir(train_human_dir)))
print('total validation horse images:', len(os.listdir(validation_horse_dir)))
print('total validation human images:', len(os.listdir(validation_human_dir)))


######### NN design #################
model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    # The second convolution
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # The third convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # The fourth convolution
    # tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    # tf.keras.layers.MaxPooling2D(2,2),

    # The fifth convolution
    # tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    # tf.keras.layers.MaxPooling2D(2,2),

    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),

    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),

    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

from tensorflow.keras.optimizers import RMSprop

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['accuracy'])


################# image preprocessing #############
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1/255)
validation_datagen = ImageDataGenerator(rescale=1/255)

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        f'{os.getcwd()}\\tmp\\horse-or-human\\',  # This is the source directory for training images
        target_size=(150, 150),  # All images will be resized to 150x150
        batch_size=128,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

# Flow training images in batches of 128 using train_datagen generator
validation_generator = validation_datagen.flow_from_directory(
        f'{os.getcwd()}\\tmp\\validation-horse-or-human\\',  # This is the source directory for training images
        target_size=(150, 150),  # All images will be resized to 150x150
        batch_size=32,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

############# training and validation at the same time #################
history = model.fit(
      train_generator,
      steps_per_epoch=8,
      epochs=15,
      verbose=2,
      validation_data = validation_generator,
      validation_steps=8
      )

print('done')