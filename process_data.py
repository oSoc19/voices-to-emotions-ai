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

german_dict = {
    'W': 'angry',
    'L': 'calm',
    'E': 'disgust',
    'A': 'fearful',
    'F': 'happy',
    'T': 'sad',
    'N': 'neutral'
}

intensity_dict = {
    '01': 'normal',
    '02': 'strong'
}

statement_dict = {
    '01': 'Kids are talking by the door',
    '02': 'Dogs are sitting by the door'
}

german_statement_dict = {
    'a01':	'Der Lappen liegt auf dem Eisschrank.',
    'a02':	'Das will sie am Mittwoch abgeben.',
    'a04':	'Heute abend könnte ich es ihm sagen.',
    'a05':	'Das schwarze Stück Papier befindet sich da oben neben dem Holzstück.',
    'a07':	'In sieben Stunden wird es soweit sein.',
    'b01':	'Was sind denn das für Tüten, die da unter dem Tisch stehen?',
    'b02':	'Sie haben es gerade hochgetragen und jetzt gehen sie wieder runter.',
    'b03':	'An den Wochenenden bin ich jetzt immer nach Hause gefahren und habe Agnes besucht.',
    'b09':	'Ich will das eben wegbringen und dann mit Karl was trinken gehen.',
    'b10':	'Die wird auf dem Platz sein, wo wir sie immer hinlegen.'
}

data_index = []
german_genders = {
    '03': 'M',
    '08': 'F',
    '09': 'F',
    '10': 'M',
    '11': 'M',
    '12': 'M',
    '13': 'F',
    '14': 'F',
    '15': 'M',
    '16': 'F'
}
def add_german_entry(file_path):
    print('Add German Index Entry: ' + file_path)
    file_name = os.path.basename(file_path).replace('.png', '')

    actor_no = file_name[0:2]
    gender = german_genders[actor_no]

    emotion = german_dict[file_name[5:6]]

    data_index.append({
        'file_path': file_path,
        'emotion': emotion,
        'statement': german_statement_dict[file_name[2:5]],
        'gender': gender
    })

def add_index_entry(file_path):
    print('Add Index Entry: ' + file_path)
    file_name = os.path.basename(file_path).replace('.png', '').split('-')

    gender = 'F'
    if (int(file_name[6]) % 2 > 0):
        gender = 'M'

    emotion = emotion_dict[file_name[2]]

    data_index.append({
        'file_path': file_path,
        'emotion': emotion,
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
            add_german_entry(target_filepath)
            create_spectrogram(file_path)


input_dir = './data/german'
iterate_dirs(input_dir)
save_index('german_index.csv')
