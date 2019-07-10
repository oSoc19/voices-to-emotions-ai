import tensorflow as tf
import os, json


emotion_dict = {
    0: 'Neutral',
    1: 'Calm',
    2: 'Happy',
    3: 'Sad',
    4: 'Angry',
    5: 'Fearful',
    6: 'Disgust',
    7: 'Surprised'
}


def create_graph():
    # Creates graph from saved model.pb
    with tf.gfile.FastGFile(os.path.abspath('model/model.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def main():
    input_data = None
    input_data_path = os.path.abspath('data/train/01-F-DE-0ba3a8cce77ed17ed7c7d1e2bc160c34df99d772.wav.json')
    if os.path.exists(input_data_path):
        with open(input_data_path, 'r') as file:
            json_data = file.read()
            input_data = json.loads(json_data)

    create_graph()

    with tf.Session() as sess:
        print('=== Model Summary ===')
        for op in sess.graph.get_operations():
            print(str(op.name))

        input_tensor = sess.graph.get_tensor_by_name('FullyConnected/Softmax:0')
        results = sess.run(input_tensor, {'InputData/X:0': input_data})

        for x in range(0, len(results)):
            predictions = results[x]
            print('=== Prediction', str(x), '===')

            for y in range(0, len(predictions)):
                p = predictions[y]
                print(emotion_dict[y], round(p * 10000) / 100, '%')
            print('====================')


if __name__ == '__main__':
    main()
