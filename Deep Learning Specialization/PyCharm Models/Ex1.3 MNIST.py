import tensorflow as tf

mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

import numpy as np
np.set_printoptions(linewidth=200)

import matplotlib.pyplot as plt
#show image at the bottom
plt.imshow(training_images[12])
plt.show()

print("Label ID:",training_labels[12])
print("Feature vector:\n",training_images[12])

training_images  = training_images / 255.0
test_images = test_images / 255.0

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=100, verbose=2)

model.evaluate(test_images, test_labels)
print(test_labels[0])