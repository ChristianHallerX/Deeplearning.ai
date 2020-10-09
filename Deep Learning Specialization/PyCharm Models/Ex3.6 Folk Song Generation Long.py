import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np

######### workaround ##########
#config = tf.compat.v1.ConfigProto(gpu_options =
#                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
#                        )
#config.gpu_options.allow_growth = True
#session = tf.compat.v1.Session(config=config)
#tf.compat.v1.keras.backend.set_session(session)


########## Download training data ##############
import os
import wget

url = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/irish-lyrics-eof.txt'
out = f'{os.getcwd()}\\tmp\\irish-lyrics-eof.txt'

if not os.path.isfile(out):
    wget.download(url, out=out)
    print('Download Finished')
else:
    print('Download Skipped')



############### Tokenize Training Data ############
tokenizer = Tokenizer()

data = open(f'{os.getcwd()}\\tmp\\irish-lyrics-eof.txt').read()

corpus = data.lower().split("\n")

tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

print(tokenizer.word_index)
# note the word index ca. 10 times longer than last exercise
print(total_words)


############ Pre Processing ###########
# generate sequence from each line
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    # sequences always one word longer (i+1)
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        input_sequences.append(n_gram_sequence)

# find longest sequence
max_sequence_len = max([len(x) for x in input_sequences])

# pad all sequences as long as longest sequence
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

print(input_sequences)

# create predictors and label
xs, labels = input_sequences[:,:-1],input_sequences[:,-1]

# one-hot-encoding
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)



########## Design LSTM ##############

# design neural network LSTM
model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len-1)) # much larger vector dimension
model.add(Bidirectional(LSTM(150))) # much larger LSTM size
model.add(Dense(total_words, activation='softmax'))

# custom learning rate
adam = Adam(lr=0.01)

model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
#earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')

model.summary()



########### Train #############
history = model.fit(xs, ys, epochs=50, verbose=0)



########## Plot ###########

import matplotlib.pyplot as plt

def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.show()

plot_graphs(history, 'accuracy')



########## Predict text ###########

# starter words for the prediction
seed_text = "I've got a bad feeling about this"

# how many words to predict after seed
next_words = 100

# a loop that predicts as many words as specified above
for _ in range(next_words):
    # turn seed into sequence
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    # pad seed sequence
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')

    # predict next token
    predicted = model.predict_classes(token_list, verbose=0)

    # turn token prediction back into word by going through word index
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    # add predicted word to seed
    seed_text += " " + output_word

print(seed_text)