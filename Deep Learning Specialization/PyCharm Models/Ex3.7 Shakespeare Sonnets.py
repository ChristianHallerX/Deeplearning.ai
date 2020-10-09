from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
### YOUR CODE HERE
#from tensorflow.keras.regularizers import l2
# Figure out how to import regularizers
###
import tensorflow.keras.utils as ku
import numpy as np
import tensorflow as tf


######### workaround ##########
#config = tf.compat.v1.ConfigProto(gpu_options =
#                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
#                        )
#config.gpu_options.allow_growth = True
#session = tf.compat.v1.Session(config=config)
#tf.compat.v1.keras.backend.set_session(session)


######## Download data ##########
import os
import wget

url = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sonnets.txt'
out = f'{os.getcwd()}\\tmp\\sonnets.txt'

if not os.path.isfile(out):
    wget.download(url, out=out)


####### Open and Tokenize ##############

tokenizer = Tokenizer()

data = open(f'{os.getcwd()}\\tmp\\sonnets.txt').read()

corpus = data.lower().split("\n")

tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1 # one more for OOV

print("Vocabulary length: ",total_words)



########## Create Token Sequences #########
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# pad sequences
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# create predictors and label
predictors, label = input_sequences[:,:-1],input_sequences[:,-1]

# one-hot-encoding of label tokens
label = ku.to_categorical(label, num_classes=total_words)



############ Design LSTM ##########

model = Sequential()
model.add(Embedding(total_words, 64, input_length=max_sequence_len-1)) # Your Embedding Layer
model.add(Bidirectional(LSTM(150, return_sequences = True))) # An LSTM Layer
model.add(Dropout(.2)) # A dropout layer
model.add(Bidirectional(LSTM(150))) # Another LSTM Layer
model.add(Dense(total_words/2,
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(0.01))) # A Dense Layer including regularizers
model.add(Dense(total_words, activation='softmax')) # A Dense Layer

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # Pick a loss function and an optimizer

model.summary()



############### Train ###########
history = model.fit(predictors, label, epochs=50, verbose=2)


############## Plot History ############

import matplotlib.pyplot as plt
acc = history.history['accuracy']
loss = history.history['loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.title('Training accuracy')

plt.figure()

plt.plot(epochs, loss, 'b', label='Training Loss')
plt.title('Training loss')
plt.legend()
plt.show()


################ Predict ##############

seed_text = "Help me Obi Wan Kenobi, you're my only hope"
next_words = 100

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted = model.predict_classes(token_list, verbose=0)
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    seed_text += " " + output_word
print(seed_text)