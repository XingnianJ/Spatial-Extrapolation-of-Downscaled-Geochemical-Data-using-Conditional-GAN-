from os import listdir
import numpy as np
import tensorflow as tf
from PIL import Image as image
def make_tfrecords(self):
    self.data_path = '/training area/images after matching/'  or '/verification area/images after matching/'
    self.save_path = '/training area/tfrecords file/'  or '/verification area/tfrecords file/'

    data_path = self.data_path
    target_path = self.save_path
    sample_list = listdir(data_path)
    for sample in sample_list:
        data_layers = listdir(data_path + '\\' + sample)
        i = 0
        for data_layer in data_layers:
            if i == 0:
                data = np.array(image.open(data_path + '\\' + sample + '\\' + data_layer))
                data = np.reshape(data, (data.shape[0], data.shape[1], 1))
                i += 1
            else:
                temp = np.array(image.open(data_path + '\\' + sample + '\\' + data_layer), dtype=np.float32)
                temp = np.reshape(temp, (temp.shape[0], temp.shape[1], 1))
                data = np.concatenate((data, temp), axis=2)
                # np.save(r"{0}\npy_files\{1}.npy".format(data_path,sample),data)
        tf_writer = tf.python_io.TFRecordWriter('{0}\\{1}.tfrecords'.format(target_path, sample))
        data = data.astype(np.float32)
        data_bytes = data.tostring()
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'image_data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data_bytes])),
                }
            ))
        tf_writer.write(example.SerializeToString())
        tf_writer.close()