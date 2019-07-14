import numpy as np
import os, librosa, gc, math
import matplotlib.pyplot as plt
import librosa.display as ldisplay

rounded_len = 100


def generate_spectrogram(file_path, output_dir):
    plt.interactive(False)

    clip, sample_rate = librosa.load(file_path, sr=16000)
    trimmed_clips = librosa.effects.split(clip, top_db=50)
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
        db_matrix = librosa.power_to_db(S, ref=np.max)
        _, y_len = db_matrix.shape
        splitted_spectrogram = np.array_split(db_matrix, math.ceil(y_len / rounded_len), axis=1)

        # Pad this shit (not sure if it's useful to do so though)
        for y in range(0, len(splitted_spectrogram)):
            short_db_matrix = splitted_spectrogram[y]
            short_db_matrix = np.pad(short_db_matrix, ((0, 0), (0, rounded_len - len(short_db_matrix[0]))),
                                     mode='reflect')

            ldisplay.specshow(short_db_matrix)

            # Save File
            plt_file_path = os.path.join(output_dir,
                                         os.path.basename(file_path) + '__' + str(x) + '__' + str(y) + '.png')
            plt.savefig(plt_file_path, dpi=400, bbox_inches='tight', pad_inches=0)
            plt.close()
            fig.clf()
            plt.close(fig)
            plt.close('all')

            print('Created spectrogram: ' + plt_file_path)
            filenames.append(plt_file_path)

    gc.collect()

    return filenames


def load_dataset(dir_path):
    dataset = []
    for p in os.listdir(dir_path):
        dataset.append(os.path.join(dir_path, p))

    return dataset


def generate_all():
    input_dir = os.path.abspath('data/automl_input')
    output_dir = os.path.abspath('data/automl_test')

    for wav in load_dataset(input_dir):
        if not wav.endswith(".wav"):
            continue

        generate_spectrogram(wav, output_dir)


# generate_all()
generate_spectrogram(os.path.abspath('data/automl_test/happy_painter.mp3'), os.path.abspath('data/automl_test'))
