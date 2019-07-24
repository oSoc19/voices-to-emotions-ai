from __future__ import division, print_function, absolute_import
from random import shuffle
import tflearn, gc, lstm_speech_data, os, math
import numpy as np

data_dir = os.path.abspath('./data')

learning_rate = 0.0001
training_epochs = 25000
batch_size = 100000
model_path = 'checkpoint/'
dropout = 0.7
lstm_units = 512

mfcc_features = 14
height = 200
classes = 8
dataset = np.array(lstm_speech_data.load_dataset(os.path.join(data_dir, 'train_noisy')))

train_dataset = dataset[:round(len(dataset) * .8)]
validate_dataset = dataset[round(len(dataset) * .8):]

shuffle(train_dataset)
shuffle(validate_dataset)

print('Loading Training Data...')
trainX, trainY = lstm_speech_data.mfcc_get_batch(train_dataset, batch_size=math.ceil(batch_size * 0.8),
                                                 mfcc_features=mfcc_features, height=height)

print('Loading Validation Data...')
testX, testY = lstm_speech_data.mfcc_get_batch(validate_dataset, batch_size=math.ceil(batch_size * 0.2),
                                               mfcc_features=mfcc_features, height=height)

acc = tflearn.metrics.accuracy()

# Network building
net = tflearn.input_data([None, mfcc_features, height])
net = tflearn.lstm(net, lstm_units, dropout=dropout)
net = tflearn.fully_connected(net, classes, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy',
                         metric=acc)

checkpoint_dir = os.path.abspath(model_path + str(learning_rate) + '-' + str(lstm_units) + '-' + str(dropout) + '-' + str(
    mfcc_features) + '-' + str(height) + '/')
checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint')
best_checkpoint_path = os.path.join(checkpoint_dir, 'best_checkpoint')

if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)

# Training
model = tflearn.DNN(net, tensorboard_verbose=1,
                    checkpoint_path=checkpoint_path, best_checkpoint_path=best_checkpoint_path, best_val_accuracy=0.75)

#  model.load(checkpoint_path)

model.fit(trainX, trainY, n_epoch=training_epochs, validation_set=(testX, testY), show_metric=True,
          batch_size=batch_size)
