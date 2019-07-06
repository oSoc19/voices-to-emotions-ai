from filehash import filehash
from shutil import copyfile
import os

# FileFormat: {Emotion}-{Gender}-{Language}-{Hash}.{wav|mp3}
# Example: 04-F-EN-91d50642dd930e9542c39d36f0516d45f4e1af0d.wav

target_dir = os.path.abspath('./data/train')

emotion_dict = {
    'Neutral': '01',
    'Calm': '02',
    'Happy': '03',
    'Sad': '04',
    'Angry': '05',
    'Fearful': '06',
    'Disgust': '07',
    'Surprised': '08'
}

language_dict = {
    'English': 'EN',
    'German': 'DE'
}

gender_dict = {
    'Male': 'M',
    'Female': 'F'
}


def get_files_recursive(dir_name, files=[]):
    dir_name = os.path.abspath(dir_name)
    dir_content = os.listdir(dir_name)

    for file_name in dir_content:
        file_path = os.path.join(dir_name, file_name)

        if (os.path.isdir(file_path)):
            files = get_files_recursive(file_path, files)

        else:
            files.append(file_path)

    return files


def rename_german_data():
    dir_path = os.path.abspath('./data/german')

    german_emotions = {
        'W': 'Angry',
        'L': 'Calm',
        'E': 'Disgust',
        'A': 'Fearful',
        'F': 'Happy',
        'T': 'Sad',
        'N': 'Neutral'
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

    for file_path in get_files_recursive(dir_path):
        file_name = os.path.basename(file_path)

        if file_name.endswith('.wav'):
            emotion = emotion_dict[german_emotions[file_name[5:6]]]
            actor_no = file_name[0:2]
            gender = german_genders[actor_no]
            file_hash = filehash(file_path)

            dst_file_name = emotion + '-' + gender + '-' + 'DE' + '-' + file_hash + '.wav'
            dst_path = os.path.join(target_dir, dst_file_name)
            copyfile(file_path, dst_path)


def rename_engie_data():
    dir_path = os.path.abspath('./data/engie')

    for file_path in get_files_recursive(dir_path):
        file_name = os.path.basename(file_path)

        if file_name.endswith('.wav'):
            file_name_parts = file_name.replace('.wav', '').split('-')

            gender = 'F'
            if (int(file_name_parts[6]) % 2 > 0):
                gender = 'M'

            emotion = file_name_parts[2]
            file_hash = filehash(file_path)
            dst_file_name = emotion + '-' + gender + '-' + 'EN' + '-' + file_hash + '.wav'
            dst_path = os.path.join(target_dir, dst_file_name)
            copyfile(file_path, dst_path)

# rename_german_data()
# rename_engie_data()
