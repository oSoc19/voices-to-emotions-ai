from __future__ import division, print_function, absolute_import
import tflearn
import lstm_speech_data
import numpy as np
import gc
import os

emotion_dict = {
    0: 'Neutral',
    1: 'Calm',
    2: 'Happy',
    3: 'Sad',
    4: 'Angry',
    5: 'Fearful',
    6: 'Disgust',
    7: 'Surprised'
}

learning_rate = 0.0001
training_epochs = 25
batch_size = 128
model_path = 'model-lstm.tflearn'

width = 20  # mfcc features
height = 500  # (max) length of utterance
classes = 8

dataset_folder = os.path.abspath('./data/test')
dataset = lstm_speech_data.load_dataset(dataset_folder=dataset_folder)

# Network building
net = tflearn.input_data([None, width, height])
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, classes, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy')


# Load model
print('Loading model...')
model = tflearn.DNN(net, tensorboard_verbose=1)
model.load(model_path)

gc.collect()

print('Evaluating model...')
evalX, evalY = lstm_speech_data.mfcc_get_batch(dataset, batch_size=batch_size, dataset_folder=dataset_folder)
predictions = model.predict(evalX)
accuracy = 0
for prediction, actual in zip(predictions, evalY):
    predicted_class = np.argmax(prediction)
    actual_class = np.argmax(actual)

    print('emotion percentages:')
    for i in range(0, len(prediction)):
        p = prediction[i]
        print(emotion_dict[i] + ': ' + str(round(p * 10000) / 100) + '%')
    print('Predicted:', emotion_dict[predicted_class], ',', 'Actual:', emotion_dict[actual_class])

    if (predicted_class == actual_class):
        accuracy += 1

accuracy = accuracy / len(evalY)

print("AVG Model Accuracy:", str(round(accuracy * 10000) / 100), '%')

gc.collect()
