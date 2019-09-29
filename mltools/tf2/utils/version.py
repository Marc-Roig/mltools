import tensorflow as tf
import os

def is_tf2():
    return tf.__version__.startswith('2.')

def filter_info_logs():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

def filter_warnings_logs():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def filter_error_logs():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)