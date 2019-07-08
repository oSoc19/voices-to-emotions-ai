from __future__ import division, print_function, absolute_import
import tflearn
import lstm_speech_data
import numpy as np
import gc
from random import shuffle

learning_rate = 0.0001
training_epochs = 25
batch_size = 256
model_path = 'model-lstm.tflearn'

mfcc_features = 20  # mfcc features
height = 500  # (max) length of utterance
classes = 8
dataset = lstm_speech_data.load_dataset()

# Network building
net = tflearn.input_data([None, mfcc_features, height])
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, classes, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, tensorboard_verbose=1)

# Load previous model to improve training
print('Loading model...')
model.load(model_path)

gc.collect()

# Train model
try:
    while True:
        print('Loading data...')

        shuffle(dataset)
        trainX, trainY = lstm_speech_data.mfcc_get_batch(dataset, batch_size=batch_size, mfcc_features=mfcc_features)

        shuffle(dataset)
        testX, testY = lstm_speech_data.mfcc_get_batch(dataset, batch_size=batch_size, mfcc_features=mfcc_features)

        model.fit(trainX, trainY, n_epoch=training_epochs, validation_set=(testX, testY), show_metric=True,
                  batch_size=batch_size)

        # Save model
        print('Saving model...')
        model.save(model_path)

        gc.collect()

except KeyboardInterrupt:
    print("KeyboardInterrupt has been caught.")

# Evaluate model
print('Evaluating model...')

shuffle(dataset)
evalX, evalY = lstm_speech_data.mfcc_get_batch(dataset, batch_size=batch_size, mfcc_features=mfcc_features)

predictions = model.predict(evalX)

accuracy = 0
for prediction, actual in zip(predictions, evalY):
    predicted_class = np.argmax(prediction)
    actual_class = np.argmax(actual)
    if (predicted_class == actual_class):
        accuracy += 1

accuracy = accuracy / len(evalY)

print("AVG Model Accuracy:", str(round(accuracy * 10000) / 100), '%')
