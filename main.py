import pylab
import wave
import os
import pandas

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
    file_name = os.path.basename(file_path).replace('.png', '').split('-')

    gender = 'F'
    if (int(file_name[6]) % 2 > 0):
        gender = 'M'

    data_index.append({
        'file_path': file_path,
        'emotion': emotion_dict[file_name[2]],
        'intensity': intensity_dict[file_name[3]],
        'statement': statement_dict[file_name[4]],
        'gender': gender
    })


def save_index(file_path):
    df = pandas.DataFrame(data_index)
    df.to_csv(file_path)


def graph_spectrogram(filename):
    sound_info, frame_rate = get_wav_info(filename)
    pylab.figure(num=None, figsize=(19, 12))
    pylab.subplot(111)
    pylab.specgram(sound_info, Fs=frame_rate)
    pylab.savefig(filename.replace('.wav', '.png'))


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

            if (os.path.isfile(target_filepath) == False):
                graph_spectrogram(file_path)


input_dir = './data/'
iterate_dirs(input_dir)
save_index('index.csv')
