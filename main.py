import pylab
import wave
import os

def get_wav_info(filename):
    wav = wave.open(filename, 'r')
    frames = wav.readframes(-1)
    sound_info = pylab.fromstring(frames, 'int16')
    frame_rate = wav.getframerate()
    wav.close()

    return sound_info, frame_rate

def graph_spectrogram(filename, inputDir, outputDir):
    sound_info, frame_rate = get_wav_info(inputDir + filename)
    pylab.figure(num=None, figsize=(19, 12))
    pylab.subplot(111)
    pylab.specgram(sound_info, Fs=frame_rate)
    pylab.savefig(outputDir + filename.replace('.wav', '.png'))

files = os.listdir('./input_data/')
print files
graph_spectrogram('03-02-06-01-01-02-01.wav', './input_data/', './results')
