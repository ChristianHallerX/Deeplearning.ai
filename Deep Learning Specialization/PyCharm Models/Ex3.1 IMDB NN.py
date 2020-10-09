import tensorflow as tf
print(tf.__version__)
import tensorflow_datasets as tfds



######### workaround ##########
#config = tf.compat.v1.ConfigProto(gpu_options =
#                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
#                        )
#config.gpu_options.allow_growth = True
#session = tf.compat.v1.Session(config=config)
#tf.compat.v1.keras.backend.set_session(session)



########  load data ############
# info not used here
imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)


######### pre-processing data #############
import numpy as np

train_data, test_data = imdb['train'], imdb['test']

training_sentences = []
training_labels = []

testing_sentences = []
testing_labels = []

# training: split in features and labels
for s,l in train_data:
  training_sentences.append(s.numpy().decode('utf8'))
  training_labels.append(l.numpy())

# testing: split in features and labels
for s,l in test_data:
  testing_sentences.append(s.numpy().decode('utf8'))
  testing_labels.append(l.numpy())

# make an numpy array from labels
training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)
print('Done pre-processing')




############# tokenizing ##################

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size = 10000
embedding_dim = 16
max_length = 120
trunc_type='post'
oov_tok = "<OOV>"

# Train
# Step 1 instantiate tokenizer
tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)

# Step 2 fit tokenizer
tokenizer.fit_on_texts(training_sentences)

# Step 3 use fitted tokenizer to create word indexes
word_index = tokenizer.word_index

# Step 4 use word index to create number sequences
sequences = tokenizer.texts_to_sequences(training_sentences)

# Step 5 pad sequences to same length
padded = pad_sequences(sequences,maxlen=max_length, truncating=trunc_type)


# Test
# Step 6 use word index to crea sequence of number (testing)
testing_sequences = tokenizer.texts_to_sequences(testing_sentences)

# Step 7 pad sequences to same length (testing)
testing_padded = pad_sequences(testing_sequences,maxlen=max_length)

print('Done tokenizing')



########### Original vs reverse text ############

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


print("Reverse embedded: ",decode_review(padded[3]))
print("\nOriginal: ",training_sentences[3])



############### Define NN #################

model = tf.keras.Sequential([
    # the embedding layer is specific to NLP - 16 dimensional
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    # make 1 dimensional
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(6, activation='relu'),
    # one neuron for 0-1 negative-positive review prediction
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()



################ Train ##################

num_epochs = 10
model.fit(padded, training_labels_final, epochs=num_epochs, validation_data=(testing_padded, testing_labels_final), verbose=2)




############### Projection on website prep ##########

e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape) # shape: (vocab_size, embedding_dim)

# write files containing vectors and meta (words) - formatting required by TF Embedding Projector

import io

out_v = io.open('vecs_imdb.tsv', 'w', encoding='utf-8')
out_m = io.open('meta_imdb.tsv', 'w', encoding='utf-8')

for word_num in range(1, vocab_size):
    # reverse embedding to word
    word = reverse_word_index[word_num]
    # write word, new line
    out_m.write(word + "\n")

    # retrieve vector
    embeddings = weights[word_num]
    # write tab, vector, new line
    out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")

out_v.close()
out_m.close()
print('Saved files.')



############# test ###########

sentence = "I really think this is amazing. honest."
sequence = tokenizer.texts_to_sequences([sentence])
print("I really think this is amazing. honest: ", sequence)













