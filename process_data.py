import gc
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
    'a01': 'Der Lappen liegt auf dem Eisschrank.',
    'a02': 'Das will sie am Mittwoch abgeben.',
    'a04': 'Heute abend könnte ich es ihm sagen.',
    'a05': 'Das schwarze Stück Papier befindet sich da oben neben dem Holzstück.',
    'a07': 'In sieben Stunden wird es soweit sein.',
    'b01': 'Was sind denn das für Tüten, die da unter dem Tisch stehen?',
    'b02': 'Sie haben es gerade hochgetragen und jetzt gehen sie wieder runter.',
    'b03': 'An den Wochenenden bin ich jetzt immer nach Hause gefahren und habe Agnes besucht.',
    'b09': 'Ich will das eben wegbringen und dann mit Karl was trinken gehen.',
    'b10': 'Die wird auf dem Platz sein, wo wir sie immer hinlegen.'
}

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


def add_german_entry(file_name, data_index, spectrogram_paths):
    print('Add German Index Entry: ' + file_name)

    actor_no = file_name[0:2]
    gender = german_genders[actor_no]

    emotion = german_dict[file_name[5:6]]

    for i in range(0, len(spectrogram_paths)):
        data_index.append({
            'file_path': spectrogram_paths[i],
            'emotion': emotion,
            'statement': german_statement_dict[file_name[2:5]],
            'gender': gender
        })


def add_engie_entry(file_name, data_index, spectrogram_paths):
    print('Add Index Entry: ' + file_name)
    file_name = file_name.split('-')

    gender = 'F'
    if (int(file_name[6]) % 2 > 0):
        gender = 'M'

    emotion = emotion_dict[file_name[2]]

    for i in range(0, len(spectrogram_paths)):
        data_index.append({
            'file_path': spectrogram_paths[i],
            'emotion': emotion,
            'statement': statement_dict[file_name[4]],
            'gender': gender
        })


def save_index(file_path, data_index):
    df = pandas.DataFrame(data_index)
    df.to_csv(file_path)


def create_spectrogram(filename):
    plt.interactive(False)

    clip, sample_rate = librosa.load(filename, sr=None)
    trimmed_clips = librosa.effects.split(clip, top_db=35)
    one_sec_sample_count = librosa.time_to_samples(1, sr=sample_rate)

    filenames = []
    for x in range(0, len(trimmed_clips)):
        trimmed_clip = clip[trimmed_clips[x][0]:trimmed_clips[x][1]]
        if (len(trimmed_clip) < one_sec_sample_count):
            continue

        fig = plt.figure(figsize=[0.72, 0.72])
        ax = fig.add_subplot(111)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_frame_on(False)
        S = librosa.feature.melspectrogram(y=trimmed_clip, sr=sample_rate)
        db_matrix = librosa.power_to_db(S, ref=numpy.max, top_db=55)
        ldisplay.specshow(db_matrix)

        # Save File
        plt_file_path = filename.replace('.wav', '__' + str(x) + '.png')
        plt.savefig(plt_file_path, dpi=400, bbox_inches='tight', pad_inches=0)
        plt.close()
        fig.clf()
        plt.close(fig)
        plt.close('all')

        print('Created spectrogram: ' + plt_file_path)
        filenames.append(plt_file_path)

    gc.collect()

    return filenames


def iterate_dirs(dir_name, dataset_type, index):
    files = os.listdir(dir_name)
    for i in range(0, len(files)):
        file_name = files[i]
        file_path = os.path.join(dir_name, file_name)
        if (os.path.isdir(file_path)):
            iterate_dirs(file_path, dataset_type, index)
        elif (file_path.endswith('.wav')):
            target_filepath = file_path.replace('.wav', '.png')
            file_name = os.path.basename(target_filepath).replace('.png', '')
            spectrogram_paths = create_spectrogram(file_path)

            if dataset_type == 'german':
                add_german_entry(file_name, index, spectrogram_paths)
            elif dataset_type == 'engie':
                add_engie_entry(file_name, index, spectrogram_paths)


input_dir = './data'


def main():
    process_german_data()
    process_engie_data()


def process_german_data():
    german_data_index = []
    german_data_dir = os.path.join(input_dir, 'german')
    iterate_dirs(german_data_dir, 'german', german_data_index)
    save_index('german_index.csv', german_data_index)


def process_engie_data():
    engie_data_index = []
    engie_data_dir = os.path.join(input_dir, 'engie')
    iterate_dirs(engie_data_dir, 'engie', engie_data_index)
    save_index('engie_index.csv', engie_data_index)


def test_one_image():
    files = create_spectrogram(os.path.join(input_dir, 'engie', 'Actor_01', '03-01-01-01-01-01-01.wav'))
    for i in range(0, len(files)):
        os.system('open ' + files[i])


main()
# process_german_data()
# test_one_image()
