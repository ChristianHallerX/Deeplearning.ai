import io
import os
import re
import shutil
import string
import tensorflow as tf

from datetime import datetime
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Activation, Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


########### Download Data ##########
print('Start Download')
url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

dataset = tf.keras.utils.get_file("aclImdb_v1.tar.gz", url,
                                    untar=True, cache_dir='.',
                                    cache_subdir='')
print('Download Done')


########### File handling #########
print('Start File Handling')
dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
os.listdir(dataset_dir)

train_dir = os.path.join(dataset_dir, 'train')
os.listdir(train_dir)

remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)
print('File Handling Done')


######### Create Dataset ##########
print('Start Creating Dataset')
batch_size = 1024
seed = 123
train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/train', batch_size=batch_size, validation_split=0.2,
    subset='training', seed=seed)
val_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/train', batch_size=batch_size, validation_split=0.2,
    subset='validation', seed=seed)

print('Show some labels and sentences')
for text_batch, label_batch in train_ds.take(1):
  for i in range(5):
    print(label_batch[i].numpy(), text_batch.numpy()[i])
print('Dataset Created')


###### Performance Cache ##########
print('Start Cache Tuning')
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
print('Cache Tuning Done')


######## Start Embedding ###########

# Embed a 1,000 word vocabulary into 5 dimensions.
embedding_layer = tf.keras.layers.Embedding(1000, 5)

print('Show some embeddings')
result = embedding_layer(tf.constant([1,2,3]))
result.numpy()


######### Text Standardization / Preprocessing #########
print('Start Standardization')
# Create a custom standardization function to strip HTML break tags '<br />'.
def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
  return tf.strings.regex_replace(stripped_html,
                                  '[%s]' % re.escape(string.punctuation), '')

# Vocabulary size and number of words in a sequence.
vocab_size = 10000
sequence_length = 100

# Use the text vectorization layer to normalize, split, and map strings to
# integers. Note that the layer uses the custom standardization defined above.
# Set maximum_sequence length as all samples are not of the same length.
vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=sequence_length)

# Make a text-only dataset (no labels) and call adapt to build the vocabulary.
text_ds = train_ds.map(lambda x, y: x)
vectorize_layer.adapt(text_ds)
print("Standardization done")


########## Define Model ##########

embedding_dim=16
model = Sequential([
  vectorize_layer,
  Embedding(vocab_size, embedding_dim, name="embedding"),
  GlobalAveragePooling1D(),
  Dense(16, activation='relu'),
  Dense(1)
])



######### Define Callback ##########
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")



########## Compile Model ###########
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])


######### Train Model #############
print("Start Training")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=[tensorboard_callback])
model.summary()


# tensorboard --logdir logs
#from tensorboard import program
#tb = program.TensorBoard()
#tb.configure(argv=[None, '--logdir', tracking_address])
#rl = tb.launch()


######### Save Embeddings #######
print('Saving Embeddings')
vocab = vectorize_layer.get_vocabulary()
print('vocab: ',vocab[:10])
# Get weights matrix of layer named 'embedding'
weights = model.get_layer('embedding').get_weights()[0]
print('weights shape:',weights.shape)

out_v = io.open('IMDBvecs.tsv', 'w', encoding='utf-8')
out_m = io.open('IMDBmeta.tsv', 'w', encoding='utf-8')

for num, word in enumerate(vocab):
    if num == 0: continue  # skip padding token from vocab
    vec = weights[num]
    out_m.write(word + "\n")
    out_v.write('\t'.join([str(x) for x in vec]) + "\n")
out_v.close()
out_m.close()

#try:
#    from google.colab import files
#except ImportError:
#    pass
#else:
#    files.download('IMDBvecs.tsv')
#    files.download('IMDBmeta.tsv')
