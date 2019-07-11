from pydub import AudioSegment
import random as r
import os
import hashlib


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


def mix_together(file1, file2, output_folder):
    sound1 = AudioSegment.from_file(file1)
    sound2 = AudioSegment.from_file(file2)

    lang = os.path.basename(file1).split("-")[2]

    if lang == "DE":
        combined = sound1.overlay(sound2)
    else:
        combined = sound1.overlay(sound2, gain_during_overlay=25)

    sha1 = hashlib.sha1()
    sha1.update(combined.raw_data)
    file_name = os.path.basename(file1).split(".")[0].split('-')
    file_hash = sha1.hexdigest()

    combined.export(output_folder + file_name[0] + '-' + file_name[1] + '-' + file_hash + ".wav", format='wav')


def create_noisy_data(input_folder, output_folder):
    for f in os.listdir(input_folder):
        if f.endswith(".wav"):
            print("Noisifying " + f)
            mix_together(input_folder + f, "./cut_noise/" + str(r.randint(0, 119)) + ".wav", output_folder)


train_dir = "./train/"
train_noisy_dir = "./train_noisy/"

multiplier = 2

if not os.path.exists(train_noisy_dir):
    os.makedirs(train_noisy_dir)

for i in range(0, multiplier):
    create_noisy_data(train_dir, train_noisy_dir)
