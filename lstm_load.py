import tensorflow as tf
import os

meta_path = os.path.abspath('model/model.meta')

# We import the meta graph and retrieve a Saver
saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)
