import tensorflow as tf

######### Workaround ########
#config = tf.compat.v1.ConfigProto(gpu_options =
#                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
#                        )
#config.gpu_options.allow_growth = True
#session = tf.compat.v1.Session(config=config)
#tf.compat.v1.keras.backend.set_session(session)


print('1 callback')
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.6):
      print("\nReached >60% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()



print('2 load data')
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()



print('3 preprocess data')
training_images=training_images/255.0
test_images=test_images/255.0



print('4 design network and model')
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])



print('5 start training')
model.fit(training_images, training_labels, epochs=5, callbacks=[callbacks], verbose=2)