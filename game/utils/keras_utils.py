"""

This helper module implements functions to load and save deep learning models.

"""

from keras.models import load_model
from keras.models import save_model

import h5py
import keras
import tempfile
import os


def load_model_from_hdf5_group(input_file, objects = None):
    """
    Load model from HDF5 file and read using Keras.
    """

    temp_fd, temp_fname = tempfile.mkstemp(prefix = "tmp_model_")

    try:
        os.close(temp_fd)

        root_item        = input_file.get("keras_model")
        serialized_model = h5py.File(temp_fname, "w")

        for attr_name, attr_value in root_item.attrs.items():
            serialized_model.attrs[attr_name] = attr_value

        for key in root_item.keys():
            input_file.copy(root_item.get(key), serialized_model, key)

        serialized_model.close()

        return load_model(temp_fname, custom_objects = objects)
    finally:
        os.unlink(temp_fname)

def save_model_to_hdf5_group(model, output_file):
    """
    Save full model (including optimizer state) to HDF5 file using Keras.
    """

    temp_fd, temp_fname = tempfile.mkstemp(prefix = "tmp_model_")

    try:
        os.close(temp_fd)

        save_model(model, temp_fname, save_format = "h5")

        serialized_model = h5py.File(temp_fname, "r")
        root_item        = serialized_model.get("/")

        serialized_model.copy(root_item, output_file, "keras_model")

        serialized_model.close()
    finally:
        os.unlink(temp_fname)

def set_gpu_memory_target(fraction):
    """
    Configure TensorFlow to use fraction of available GPU memory for multiple processes to run in
    parallel. For example, set memory fraction to 0.5 to run 2 worker processes. This function has
    no effect if Keras is not using TensorFlow backend.

    By default, TensorFlow tries to map all available GPU memory in advance.

    If using Python multiprocessing, this function must be called from worker process, not parent.
    """

    if keras.backend.backend() != "tensorflow":
        return

    # Import TensorFlow modules here, not at top, in case library is not installed.
    from keras.backend.tensorflow_backend import set_session

    import tensorflow as tf

    tf_config = tf.ConfigProto()

    tf_config.gpu_options.per_process_gpu_memory_fraction = fraction

    set_session(tf.Session(config = tf_config))
