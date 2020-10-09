##### Notice: this LSTM is training very slowly. Ca. 4 min per epoch

import tensorflow as tf
print(tf.__version__)

######### workaround ##########
#config = tf.compat.v1.ConfigProto(gpu_options =
#                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
#                        )
#config.gpu_options.allow_growth = True
#session = tf.compat.v1.Session(config=config)
#tf.compat.v1.keras.backend.set_session(session)


########### Download and Load Data ###############
import tensorflow_datasets as tfds
imdb, info = tfds.load("imdb_reviews/subwords8k", with_info=True, as_supervised=True)

# separating train/test
train_data, test_data = imdb['train'], imdb['test']

# workaround for adding extra dimensions to data
train_data = train_data.map(lambda x_text, x_label: (x_text, tf.expand_dims(x_label, -1)))
test_data = test_data.map(lambda x_text, x_label: (x_text, tf.expand_dims(x_label, -1)))


############# Tokenizer ###############
# instantiate tokenizer with subwords settings
tokenizer = info.features['text'].encoder
print(tokenizer.subwords)

# example of subwords8k encoding

sample_string = 'TensorFlow, from basics to mastery'

tokenized_string = tokenizer.encode(sample_string)
print('Tokenized string is {}'.format(tokenized_string))

original_string = tokenizer.decode(tokenized_string)
print('The original string: {}'.format(original_string))

for ts in tokenized_string:
  print ('{} ----> {}'.format(ts, tokenizer.decode([ts])))


#######  Pre-processing ######

BUFFER_SIZE = 10000
BATCH_SIZE = 64

train_dataset = train_data.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.padded_batch(BATCH_SIZE, tf.compat.v1.data.get_output_shapes(train_dataset))

test_dataset = test_data.padded_batch(BATCH_SIZE, tf.compat.v1.data.get_output_shapes(test_data))


###### Design NN ##########
embedding_dim = 64
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(tokenizer.vocab_size, embedding_dim),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()


####### Train ##########
# start training. subwords are much slower in training than full words.
num_epochs = 2
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
history = model.fit(train_data, epochs=num_epochs, validation_data=test_data, verbose=1)


####### Plot History ######
import matplotlib.pyplot as plt


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()

plot_graphs(history, "acc")
plot_graphs(history, "loss")


