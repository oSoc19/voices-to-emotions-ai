from __future__ import division, print_function, absolute_import
from random import shuffle
import tflearn, gc, sys, lstm_speech_data
import tensorflow as tf

learning_rate = 0.0001
training_epochs = 50
batch_size = 1000
model_path = 'model/model'

mfcc_features = 20  # mfcc features
height = 500  # (max) length of utterance
classes = 8
dataset = lstm_speech_data.load_dataset()

# Network building
net = tflearn.input_data([None, mfcc_features, height])
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, classes, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy')

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
sess = tf.Session(config=config)

# Training
model = tflearn.DNN(net, tensorboard_verbose=1, session=sess)

# Load previous model to improve training
print('Loading model...')
model.load(model_path)

gc.collect()

# Train model
while True:
    try:
        print('Loading Training Data...')
        shuffle(dataset)
        trainX, trainY = lstm_speech_data.mfcc_get_batch(dataset, batch_size=batch_size, mfcc_features=mfcc_features)

        print('Loading Validation Data...')
        shuffle(dataset)
        testX, testY = lstm_speech_data.mfcc_get_batch(dataset, batch_size=batch_size, mfcc_features=mfcc_features)

        model.fit(trainX, trainY, n_epoch=training_epochs, validation_set=(testX, testY), show_metric=True,
                  batch_size=batch_size)

        # Save model
        print('Saving model...')

        del tf.get_collection_ref(tf.GraphKeys.TRAIN_OPS)[:]
        model.save(model_path)

        gc.collect()

    except KeyboardInterrupt:
        print("KeyboardInterrupt has been caught.")
        break

    except:
        e = sys.exc_info()[0]
        print(e)
