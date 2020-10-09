import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
import wget


######### workaround ##########
#config = tf.compat.v1.ConfigProto(gpu_options =
#                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
#                        )
#config.gpu_options.allow_growth = True
#session = tf.compat.v1.Session(config=config)
#tf.compat.v1.keras.backend.set_session(session)



# Import the inception model
from tensorflow.keras.applications.inception_v3 import InceptionV3

# Create an instance of the inception model from the local pre-trained weights
local_weights_file = f"{os.getcwd()}\\tmp\\inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"


pre_trained_model = InceptionV3(input_shape=(150, 150, 3),
                                include_top=False,
                                weights='imagenet')  # Your Code Here

pre_trained_model.load_weights(local_weights_file)

# Make all the layers in the pre-trained model non-trainable
for layer in pre_trained_model.layers:
    layer.trainable = False  # Make all the layers non-trainable
print('Loaded InceptionV3')
# summary of original model
#pre_trained_model.summary()



########### grab last layer ##############
last_layer = pre_trained_model.get_layer('mixed7') # Your Code Here
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output # Your Code Here



########### Our own trainable NN at the end ##########
from tensorflow.keras.optimizers import RMSprop

# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)
# Add a final sigmoid layer for classification
x = layers.Dense  (1, activation='sigmoid')(x)

model = Model(pre_trained_model.input, x)

model.compile(optimizer = RMSprop(lr=0.0001),
              loss = 'binary_crossentropy',
              metrics = ['acc'])
# summary of model including our stuff
#model.summary()




############ Download and Unzip ################
url = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip'
out = f'{os.getcwd()}\\tmp\\horse-or-human.zip'

if not os.path.isfile(out):
    wget.download(url, out=out)
    print('Download1 Finished')
else:
    print('Download1 Skipped')

url = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/validation-horse-or-human.zip'
out2 = f'{os.getcwd()}\\tmp\\validation-horse-or-human.zip'

if not os.path.isfile(out):
    wget.download(url, out=out2)
    print('Download2 Finished')
else:
    print('Download2 Skipped')


import zipfile

zip_ref = zipfile.ZipFile(out, 'r')
zip_ref.extractall(f'{os.getcwd()}\\tmp\\training')
zip_ref.close()

zip_ref = zipfile.ZipFile(out2, 'r')
zip_ref.extractall(f'{os.getcwd()}\\tmp\\validation')
zip_ref.close()
print('Finished Unzipping')




######### Setup directories ###########
train_dir = f'{os.getcwd()}\\tmp\\training'
validation_dir = f'{os.getcwd()}\\tmp\\validation'

train_horses_dir = f'{os.getcwd()}\\tmp\\training\\horses' # Your Code Here
train_humans_dir = f'{os.getcwd()}\\tmp\\training\\humans' # Your Code Here
validation_horses_dir = f'{os.getcwd()}\\tmp\\validation\\horses' # Your Code Here
validation_humans_dir = f'{os.getcwd()}\\tmp\\validation\\humans' # Your Code Here

train_horses_fnames = os.listdir(train_horses_dir) # Your Code Here
train_humans_fnames = os.listdir(train_humans_dir) # Your Code Here
validation_horses_fnames = os.listdir(validation_horses_dir) # Your Code Here
validation_humans_fnames = os.listdir(validation_humans_dir) # Your Code Here

print('train horses: ',len(train_horses_fnames)) # Your Code Here
print('train humans: ',len(train_humans_fnames)) # Your Code Here
print('validation horses: ',len(validation_horses_fnames)) # Your Code Here
print('validation humans: ',len(validation_humans_fnames)) # Your Code Here




############ Image pre-processing #############
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')


# Note that the validation data should NOT be augmented!
validation_datagen = ImageDataGenerator(
    rescale=1./255)

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
    train_dir,  # This is the source directory for training images
    target_size=(150, 150),  # All images will be resized to 150x150
    batch_size=20,
    class_mode='binary') # Since we use binary_crossentropy loss, we need binary labels

# Flow validation images in batches of 20 using test_datagen generator
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,  # This is the source directory for validation images
    target_size=(150, 150),  # All images will be resized to 150x150
    batch_size=20,
    class_mode='binary') # Since we use binary_crossentropy loss, we need binary labels
print('Done pre-processing')


############# Callback #############
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('acc') > 0.97):
            print("\nReached 97% accuracy so cancelling training!")
            self.model.stop_training = True

callbacks = myCallback()



############# Start Training ############
history = model.fit_generator(train_generator,
                              steps_per_epoch=8,
                              epochs=3,
                              verbose=2,
                              validation_data=validation_generator,
                              validation_steps=8,
                              callbacks=[callbacks])


############## Plot History ###############
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()
plt.show()