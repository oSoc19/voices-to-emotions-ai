from __future__ import division, print_function, absolute_import
from random import shuffle
import tflearn, gc, lstm_speech_data, os, math, sherpa

data_dir = os.path.abspath('./data')

learning_rate = 0.0001
training_epochs = 25
batch_size = 100
model_path = 'checkpoint/'
dropout = 0.9
lstm_units = 128

mfcc_features = 14
height = 200
classes = 8
train_dataset = lstm_speech_data.load_dataset(os.path.join(data_dir, 'train')) + lstm_speech_data.load_dataset(
    os.path.join(data_dir, 'train_noisy'))

acc = tflearn.metrics.accuracy()

# Network building
net = tflearn.input_data([None, mfcc_features, height])
net = tflearn.lstm(net, lstm_units, dropout=dropout)
net = tflearn.fully_connected(net, classes, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy',
                         metric=acc)

checkpoint_path = model_path + str(learning_rate) + '-' + str(lstm_units) + '-' + str(dropout) + '-' + str(
    mfcc_features) + '-' + str(height) + '/' + 'checkpoint'

# Training
model = tflearn.DNN(net, tensorboard_verbose=1,
                    checkpoint_path=checkpoint_path)

print('Loading Training Data...')
shuffle(train_dataset)
trainX, trainY = lstm_speech_data.mfcc_get_batch(train_dataset, batch_size=math.ceil(batch_size * 0.8),
                                                 mfcc_features=mfcc_features, height=height)

print('Loading Validation Data...')
shuffle(train_dataset)
testX, testY = lstm_speech_data.mfcc_get_batch(train_dataset, batch_size=math.ceil(batch_size * 0.2),
                                               mfcc_features=mfcc_features, height=height)

model.fit(trainX, trainY, n_epoch=training_epochs, validation_set=(testX, testY), show_metric=True,
          batch_size=batch_size)

gc.collect()
