from __future__ import division, print_function, absolute_import
import tflearn
import lstm_speech_data
import numpy as np

learning_rate = 0.0001
training_iters = 50
training_epochs = 10
batch_size = 128

width = 20  # mfcc features
height = 500  # (max) length of utterance
classes = 7

batch = word_batch = lstm_speech_data.mfcc_batch_generator(batch_size)
X, Y = next(batch)
trainX, trainY = X, Y
testX, testY = X, Y

# Network building
net = tflearn.input_data([None, width, height])
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, classes, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, tensorboard_verbose=0)
for i in range(0, training_iters):
    model.fit(trainX, trainY, n_epoch=training_epochs, validation_set=(testX, testY), show_metric=True,
              batch_size=batch_size)
    _y = model.predict(X)

# Save model
model.save("tflearn.lstm.model")

# Print da statistics
# print(Y)
avg_acc_results = np.average(_y, axis=0)
for i in range(0, len(avg_acc_results)):
    print('Accuracy of #' + str(i) + ": " + str(round(avg_acc_results[i] * 100)) + '%')
