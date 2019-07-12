import os, shutil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


def freeze_graph(model_folder, output_folder):
    # We retrieve our checkpoint fullpath
    try:
        checkpoint = tf.train.get_checkpoint_state(model_folder)
        input_checkpoint = checkpoint.model_checkpoint_path
        print("[INFO] input_checkpoint:", input_checkpoint)
    except:
        input_checkpoint = model_folder
        print("[INFO] Model folder", model_folder)

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We import the meta graph and retrieve a Saver
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

    # We start a session and restore the graph weights
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)

        input_tensor = sess.graph.get_tensor_by_name("InputData/X:0")
        output_tensor = sess.graph.get_tensor_by_name("FullyConnected/Softmax:0")

        builder = tf.saved_model.builder.SavedModelBuilder(output_folder)
        prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs={
                "mfcc": tf.saved_model.utils.build_tensor_info(input_tensor)
            },
            outputs={
                "emotions": tf.saved_model.utils.build_tensor_info(output_tensor)
            },
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME,
        )

        tensor_info_x = tf.saved_model.utils.build_tensor_info(input_tensor)
        tensor_info_y = tf.saved_model.utils.build_tensor_info(output_tensor)

        classification_signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs={tf.saved_model.signature_constants.CLASSIFY_INPUTS: tensor_info_x},
            outputs={
                tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES: tensor_info_y
            },
            method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME,
        )

        builder.add_meta_graph_and_variables(
            sess,
            [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                "predict_emotions": prediction_signature,
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: classification_signature,
            },
            main_op=tf.tables_initializer(),
        )
        builder.save()

        print("[INFO] output_graph:", model_folder)
        print("[INFO] all done")


if __name__ == '__main__':
    input_folder = os.path.abspath('model')
    output_folder = os.path.abspath('output')

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    freeze_graph(input_folder, output_folder)
