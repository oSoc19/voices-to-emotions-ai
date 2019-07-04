import wave
import os
import pandas
import matplotlib.pyplot as plt
import librosa
import librosa.display as ldisplay
import numpy

emotion_dict = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

intensity_dict = {
    '01': 'normal',
    '02': 'strong'
}

statement_dict = {
    '01': 'Kids are talking by the door',
    '02': 'Dogs are sitting by the door'
}

good_emotions = frozenset(['neutral', 'calm', 'happy'])

data_index = []


def get_wav_info(filename):
    print('Reading: ' + filename)
    wav = wave.open(filename, 'r')
    frames = wav.readframes(-1)
    sound_info = pylab.fromstring(frames, 'int16')
    frame_rate = wav.getframerate()
    wav.close()

    return sound_info, frame_rate


def add_index_entry(file_path):
    print('Add Index Entry: ' + file_path)
    file_name = os.path.basename(file_path).replace('.png', '').split('-')

    gender = 'F'
    if (int(file_name[6]) % 2 > 0):
        gender = 'M'

    emotion = emotion_dict[file_name[2]]

    emotion_cat = 'bad'
    if (emotion in good_emotions):
        emotion_cat = 'good'

    data_index.append({
        'file_path': file_path,
        'emotion': emotion,
        'category': emotion_cat,
        'intensity': intensity_dict[file_name[3]],
        'statement': statement_dict[file_name[4]],
        'gender': gender
    })


def save_index(file_path):
    df = pandas.DataFrame(data_index)
    df.to_csv(file_path)


def create_spectrogram(filename):
    plt.interactive(False)
    clip, sample_rate = librosa.load(filename, sr=None)
    fig = plt.figure(figsize=[0.72, 0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    ldisplay.specshow(librosa.power_to_db(S, ref=numpy.max))
    filename = filename.replace('.wav', '.png')
    plt.savefig(filename, dpi=400, bbox_inches='tight', pad_inches=0)
    plt.close()
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del filename, clip, sample_rate, fig, ax, S


def iterate_dirs(dir_name):
    files = os.listdir(dir_name)
    for i in range(0, len(files)):
        file_name = files[i]
        file_path = os.path.join(dir_name, file_name)
        if (os.path.isdir(file_path)):
            iterate_dirs(file_path)
        elif (file_path.endswith('.wav')):
            target_filepath = file_path.replace('.wav', '.png')
            add_index_entry(target_filepath)
            create_spectrogram(file_path)


input_dir = './data/'
iterate_dirs(input_dir)
save_index('index.csv')
