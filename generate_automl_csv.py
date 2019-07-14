import os
import pandas as pd

output_dir = os.path.abspath('data/automl_output')
library = []

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

dircontent = os.listdir(output_dir)
for file in dircontent:
    if not file.endswith('.png'):
        continue

    file_parts = file.split('-')
    emotion = emotion_dict[file_parts[0]]
    gender = file_parts[1]

    if gender == 'M':
        gender = 'male'
    elif gender == 'F':
        gender = 'female'
    else:
        continue

    library.append({
        'gs': 'gs://voices-to-emotions-vcm/spectrograms/' + file,
        'emotion': emotion,
        'gender': gender
    })

df = pd.DataFrame(library)
df.to_csv(os.path.join(output_dir, 'index.csv'), encoding='utf-8', index=False, header=False,
          columns=['gs', 'emotion', 'gender'])
