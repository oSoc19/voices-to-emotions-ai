from pydub import AudioSegment
import random as r
import os


def split_audio_file():
    fullAudio = AudioSegment.from_wav("./bg_noise.wav")

    t1 = 0
    t2 = 30000

    teller = 1
    while t2 < 3600000:
        newAudio = fullAudio[t1:t2]
        t1 = teller * 30000
        t2 = (teller + 1) * 30000
        newAudio.export('./cut_noise/' + str(teller) + '.wav', format="wav")
        teller += 1

def mix_together(file1, file2, output_folder, appendix):
    sound1 = AudioSegment.from_file(file1)
    sound2 = AudioSegment.from_file(file2)

    lang = os.path.basename(file1).split("-")[2]

    if lang == "DE":
        combined = sound1.overlay(sound2)
    else:
        combined = sound1.overlay(sound2, gain_during_overlay=25)

    combined.export(output_folder + os.path.basename(file1).split(".")[0] + "_" + appendix + ".wav", format='wav')

def create_noisy_data(input_folder, output_folder, appendix):
    for f in os.listdir(input_folder):
        if f.endswith(".wav"):
            print("Noisifying " + f)
            mix_together(input_folder + f, "./cut_noise/" + str(r.randint(0, 119)) + ".wav", output_folder, appendix)

for letter in "bc":
    create_noisy_data("./train/", "./train_noisy/", letter)