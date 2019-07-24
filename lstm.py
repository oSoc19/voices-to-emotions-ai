from __future__ import division, print_function, absolute_import
from random import shuffle
import tflearn, lstm_speech_data, os, gc
import numpy as np
import tensorflow as tf

data_dir = os.path.abspath('./data')

learning_rate = 0.0001
training_epochs = 100
batch_size = 100000
dropout = 0.65
lstm_units = 256

mfcc_features = 14
height = 200
classes = 8
dataset = np.array(lstm_speech_data.load_dataset(os.path.join(data_dir, 'train_noisy')))

shuffle(dataset)

train_dataset = dataset[:round(len(dataset) * .8)]
validate_dataset = dataset[round(len(dataset) * .8):]

print('Loading Training Data...')
trainX, trainY = lstm_speech_data.mfcc_get_batch(train_dataset, mfcc_features=mfcc_features, height=height)

print('Loading Validation Data...')
testX, testY = lstm_speech_data.mfcc_get_batch(validate_dataset, mfcc_features=mfcc_features, height=height)

checkpoint_key = str(learning_rate) + '-' + str(lstm_units) + '-' + str(dropout) + '-' + str(
    mfcc_features) + '-' + str(height)

checkpoint_dir = os.path.abspath(os.path.join('checkpoint', checkpoint_key))
best_checkpoint_dir = os.path.abspath(os.path.join('best_checkpoint', checkpoint_key))

if not os.path.exists(os.path.dirname(checkpoint_dir)):
    os.mkdir(os.path.dirname(checkpoint_dir))

if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)

if not os.path.exists(os.path.dirname(best_checkpoint_dir)):
    os.mkdir(os.path.dirname(best_checkpoint_dir))

if not os.path.exists(best_checkpoint_dir):
    os.mkdir(best_checkpoint_dir)

checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint-save')
best_checkpoint_path = os.path.join(best_checkpoint_dir, 'checkpoint-save')

# Network building
net = tflearn.input_data([None, mfcc_features, height])
net = tflearn.lstm(net, lstm_units, dropout=dropout)
net = tflearn.fully_connected(net, classes, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy')


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
sess = tf.Session(config=config)

# Training
model = tflearn.DNN(net, tensorboard_verbose=1, best_checkpoint_path=best_checkpoint_path, best_val_accuracy=0.6, session=sess)

print('Loading checkpoint...')
model.load(checkpoint_path)

gc.collect()

for i in range(0, 100000000000):
    shuffle(train_dataset)
    shuffle(validate_dataset)

    model.fit(trainX, trainY, n_epoch=training_epochs, validation_set=(testX, testY), show_metric=True)

    del tf.get_collection_ref(tf.GraphKeys.TRAIN_OPS)[:]
    print('Saving Checkpoint')
    model.save(checkpoint_path)

    gc.collect()
