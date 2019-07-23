import os
import random

DIR = './train_noisy'
files = [name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]
len_files = len(files)

NEWDIR = './test_files'

names = open('test_files.txt', 'w')
for i in range(50):
    filename = files[random.randint(0, len_files - 1)]
    os.rename(os.path.join(DIR, filename), os.path.join(NEWDIR, filename))
    names.write(filename + "\n")

names.close()
