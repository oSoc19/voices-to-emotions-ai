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

mfcc_features = 64  # mfcc features
height = 500  # (max) length of utterance
classes = 8

dataset_folder = os.path.abspath('./data/test')

# Network building
net = tflearn.input_data([None, mfcc_features, height])
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, classes, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy')

# Load model
print('Loading model...')
model = tflearn.DNN(net, tensorboard_verbose=1)
model.load(model_path)

gc.collect()

print('Evaluating model...')


def evaluate_predictions(filename, predictions):
    human_assigned_class = int(os.path.basename(filename).split('-')[0]) - 1

    print('###', 'Evaluation For:', filename, '###')
    predictions_mean = np.mean(predictions, axis=0)
    predictions_count = [0] * 8
    for prediction_arr in predictions:
        max_pred = np.argmax(prediction_arr)
        predictions_count[max_pred] += 1

    predicted_class = np.argmax(predictions_count)
    for i in range(0, len(predictions_mean)):
        p = predictions_mean[i]
        print(emotion_dict[i] + ':', str(round(p * 10000) / 100), '%', ' -- ' + 'Predicted:',
              str(predictions_count[i]) + '#')

    print('Predicted Class:', emotion_dict[predicted_class], ', Actual class:', emotion_dict[human_assigned_class])

    if predicted_class == human_assigned_class:
        return True


test_audio_files = os.listdir(dataset_folder)

correct_predictions = 0
total_predictions = 0
for file in test_audio_files:
    if (file.endswith('.wav') or file.endswith('.mp3')) == False:
        continue

    total_predictions += 1

    file_path = os.path.join(dataset_folder, file)
    audio_data = lstm_speech_data.load_audio_data(file_path, mfcc_features=mfcc_features)
    predictions = model.predict(audio_data)

    if evaluate_predictions(file, predictions):
        correct_predictions += 1

print('Accuracy:', str(round((correct_predictions / total_predictions) * 10000) / 100), '%')
