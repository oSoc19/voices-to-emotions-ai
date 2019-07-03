import pylab
import wave
import os


def get_wav_info(filename):
    print('Reading: ' + filename)
    wav = wave.open(filename, 'r')
    frames = wav.readframes(-1)
    sound_info = pylab.fromstring(frames, 'int16')
    frame_rate = wav.getframerate()
    wav.close()

    return sound_info, frame_rate


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
            graph_spectrogram(file_path)


input_dir = './data/'
iterate_dirs(input_dir)
