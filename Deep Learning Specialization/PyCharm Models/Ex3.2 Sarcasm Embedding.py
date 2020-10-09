import json
import tensorflow as tf
print(tf.__version__)
import os

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences



######### workaround ##########
#config = tf.compat.v1.ConfigProto(gpu_options =
#                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
#                        )
#config.gpu_options.allow_growth = True
#session = tf.compat.v1.Session(config=config)
#tf.compat.v1.keras.backend.set_session(session)



############## Settings #####################

#vocab_size = 10000 shorter improves loss
vocab_size = 1000

embedding_dim = 16

#max_length = 100 shorter improves loss
max_length = 16

trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size = 20000




############ Download, Unzip and Load ###############
import wget
url = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sarcasm.json'
out = f'{os.getcwd()}\\tmp\\sarcasm.json'

if not os.path.isfile(out):
    wget.download(url, out=out)
    print('Download Finished')
else:
    print('Download Skipped')

# transfer from json to a sentences and labels lists
with open(out, 'r') as f:
    datastore = json.load(f)

sentences = []
labels = []

for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])

print('Finished download and loading')


######### splitting ############

training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]



############ Tokenize and Numpy ###############

# Step 1 instantiate tokenizer
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)

# Step 2 fit tokenizer
tokenizer.fit_on_texts(training_sentences)

# Step 3 use fitted tokenizer to create word indexes
word_index = tokenizer.word_index

# Step 4 use word index to create number sequences (train)
training_sequences = tokenizer.texts_to_sequences(training_sentences)

# Step 5 pad sequences to same length (train)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# Step 4 use word index to create number sequences (test)
testing_sequences = tokenizer.texts_to_sequences(testing_sentences)

# Step 5 pad sequences to same length (test)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# np arrays needed for TF2.0
import numpy as np

training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)

print('Done tokenizing')



############### Define NN ##################

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

############## Training ####################
num_epochs = 30
history = model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels), verbose=2)





########### Plot history ###################

import matplotlib.pyplot as plt


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()


plot_graphs(history, "accuracy")
plot_graphs(history, "loss")




############ reverse a sentence #############

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_sentence(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

print(decode_sentence(training_padded[2]))
print(training_sentences[2])

# 1 = sarcasm, 0 = not sarcasm
print(labels[2])


############ save projection files ###############
e = model.layers[0]
weights = e.get_weights()[0]

print(weights.shape) # shape: (vocab_size, embedding_dim)

# write files containing vectors and meta (words) - formatting required by TF Embedding Projector

import io

out_v = io.open('vecs_sarcasm.tsv', 'w', encoding='utf-8')
out_m = io.open('meta_sarcasm.tsv', 'w', encoding='utf-8')
for word_num in range(1, vocab_size):
  word = reverse_word_index[word_num]
  embeddings = weights[word_num]
  out_m.write(word + "\n")
  out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()



############ predict two sentences #############

sentence = ["granny starting to fear spiders in the garden might be real", "game of thrones season finale showing this sunday night"]
sequences = tokenizer.texts_to_sequences(sentence)
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
my_pred = model.predict(padded)
print("prediction:\ngranny starting to fear spiders in the garden might be real: {}\ngame of thrones season finale showing this sunday night: {}".format(my_pred[0],my_pred[1]))

