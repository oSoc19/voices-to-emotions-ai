import pylab
import wave
import os


def get_wav_info(filename):
    print 'Reading: ' + filename
    wav = wave.open(filename, 'r')
    frames = wav.readframes(-1)
    sound_info = pylab.fromstring(frames, 'int16')
    frame_rate = wav.getframerate()
    wav.close()

    return sound_info, frame_rate


def graph_spectrogram(filename, inputDir, outputDir):
    if (filename.endswith('.wav') == False): return

    sound_info, frame_rate = get_wav_info(inputDir + filename)
    pylab.figure(num=None, figsize=(19, 12))
    pylab.subplot(111)
    pylab.specgram(sound_info, Fs=frame_rate)
    pylab.savefig(outputDir + filename.replace('.wav', '.png'))


files = os.listdir('./input_data/')
for i in range(0, len(files)):
    file = files[i]
    graph_spectrogram(file, './input_data/', './output_data/')
