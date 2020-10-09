import json
import tensorflow as tf
print(tf.__version__)
import csv
import random
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers

######### workaround ##########
#config = tf.compat.v1.ConfigProto(gpu_options =
#                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
#                        )
#config.gpu_options.allow_growth = True
#session = tf.compat.v1.Session(config=config)
#tf.compat.v1.keras.backend.set_session(session)


########## Settings ##############
embedding_dim = 100
max_length = 16
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size=160000 #Your dataset size here. Experiment using smaller values (i.e. 16000), but don't forget to train on at least 160000 to see the best effects
test_portion=.1


########## Download training data ##############
import os
import wget

url = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/training_cleaned.csv'
out = f'{os.getcwd()}\\tmp\\twitter_training_cleaned.csv'

if not os.path.isfile(out):
    wget.download(url, out=out)
    print('Download Finished')
else:
    print('Download Skipped')


########## Loading CSV ############

num_sentences = 0
corpus = []

with open(f'{os.getcwd()}\\tmp\\twitter_training_cleaned.csv', encoding='utf8', errors='ignore') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
      # Your Code here. Create list items where the first item is the text, found in row[5], and the second is the label. Note that the label is a '0' or a '4' in the text. When it's the former, make
      # your label to be 0, otherwise 1. Keep a count of the number of sentences in num_sentences
        list_item = []
        list_item.append(row[5])
        this_label = row[0]
        if this_label == '0':
            list_item.append(0)
        else:
            list_item.append(1)
        num_sentences = num_sentences + 1
        corpus.append(list_item)

print("Number of Sentences: ",num_sentences)
print("Length of Corpus: ",len(corpus))


############## Separate Sentence and Labels ###########

sentences = []
labels = []
random.shuffle(corpus)

for x in range(training_size):
    sentences.append(corpus[x][0]) # YOUR CODE HERE
    labels.append(corpus[x][1]) # YOUR CODE HERE

print("Sentence: ",sentences[10])
print("Sentence Label: ",labels[10])


############ Tokenize Sentences ##############

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences) # YOUR CODE HERE

word_index = tokenizer.word_index
vocab_size = len(word_index) # YOUR CODE HERE

# 1 Create Sequences
sequences = tokenizer.texts_to_sequences(sentences) # YOUR CODE HERE

# 2 Pad
padded = pad_sequences(sequences) # YOUR CODE HERE

# Split
split = int(test_portion * training_size)

test_sequences = padded[0:split] # YOUR CODE HERE
training_sequences = padded[split:training_size] # YOUR CODE HERE

test_labels = labels[0:split] # YOUR CODE HERE
training_labels = labels[split:training_size] # YOUR CODE HERE

# convert all to numpy
training_sequences = np.array(training_sequences)
training_labels = np.array(training_labels)
test_sequences = np.array(test_sequences)
test_labels = np.array(test_labels)

print("Tokenized Vocabulary size: ",vocab_size)
print("Tokenized Word index of letter i: ",word_index['i'])



###### Download Pre-Trained Model ############

url = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/glove.6B.100d.txt'
out = f'{os.getcwd()}\\tmp\\glove.6B.100d.txt'

if not os.path.isfile(out):
    wget.download(url, out=out)
    print('Embeddings downloaded.')
else:
    print('Embeddings download skipped.')


########### Load Pre-Trained Model ##################
embeddings_index = {}

#with open(f'{os.getcwd()}\\tmp\\glove.6B.100d.txt', encoding='utf8', errors='ignore') as csvfile:

with open(f'{os.getcwd()}\\tmp\\glove.6B.100d.txt', encoding='utf8', errors='ignore') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

embeddings_matrix = np.zeros((vocab_size + 1, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embeddings_matrix[i] = embedding_vector



########## Design Embedding NN ###########
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size+1, output_dim=embedding_dim, input_length=max_length, weights=[embeddings_matrix], trainable=False), ### here we load up the embedding_matrix from Stanford
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv1D(64, 5, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=4),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(1, activation='sigmoid') # YOUR CODE HERE - experiment with combining different types, such as convolutions and LSTMs
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy']) # YOUR CODE HERE
model.summary()



############ Train ##################
print('Start Training')
num_epochs = 50
history = model.fit(training_sequences, training_labels, epochs=num_epochs, validation_data=(test_sequences, test_labels), verbose=2)

print("Training Complete")



############# Plotting #############
import matplotlib.image  as mpimg
import matplotlib.pyplot as plt

#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot(epochs, acc, 'r')
plt.plot(epochs, val_acc, 'b')
plt.title('Training and validation accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["Accuracy", "Validation Accuracy"])

plt.figure()
plt.show()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot(epochs, loss, 'r')
plt.plot(epochs, val_loss, 'b')
plt.title('Training and validation loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["Loss", "Validation Loss"])

plt.figure()
plt.show()