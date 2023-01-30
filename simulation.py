import json
import collections
import glob
import tensorflow as tf
import os
import math
import random
from PIL import Image as image
from imageio import imwrite
import numpy as np


def simulation(self):
    self.input_dir = '/verification area/tfrecords file/'
    self.output_dir = '/verification area/simulation results/'
    self.mode = 'test'
    self.checkpoint = 'model saved after training/'
    with open(self.checkpoint + '\\' + 'super_parameter.txt', mode='r') as f:
        super_parameter = json.load(f)
    self.ver_nor = '/verification area/images exported from ArcGIS/'
    self.variables_path =  '/verification area/images exported from ArcGIS/'

    self.freq = int(super_parameter['self.freq'])
    self.conditions = int(super_parameter['self.conditions'])
    self.targets = int(super_parameter['self.targets'])
    self.l1_weight = 1
    self.gan_weight = 2
    self.size = int(super_parameter['self.size'])
    self.result_path = self.output_path
    self.batch_size = 1
    self.seed = None
    self.ngf = int((64 / 3) * self.targets)
    self.ndf = int((64 / 6) * (self.targets + self.conditions))

    Examples = collections.namedtuple("Examples", "inputs, targets, count, steps_per_epoch")
    Model = collections.namedtuple("Model","outputs")

    def preprocess(image):
        with tf.name_scope("preprocess"):
            return image * 2 - 1

    def deprocess(image):
        with tf.name_scope("deprocess"):
            #         # [-1, 1] => [0, 1]
            return (image + 1) / 2

    def load_examples():
        if self.input_dir is None or not os.path.exists(self.input_dir):
            raise Exception("input_dir does not exist")
        input_paths = glob.glob(os.path.join(self.input_dir, "*.tfrecords"))
        with tf.name_scope("load_data"):
            path_queue = tf.train.string_input_producer(input_paths, shuffle=False)
            reader = tf.TFRecordReader()
            _, serialized_example = reader.read(path_queue)
            features = tf.parse_single_example(
                serialized_example,
                features={
                    'image_data': tf.FixedLenFeature([], tf.string),
                }
            )
            raw_input = tf.decode_raw(features['image_data'], out_type=tf.float32)
            raw_input = tf.reshape(raw_input, [self.size, self.size, self.conditions])
            targets = preprocess(tf.random_uniform([self.size, self.size, self.targets], 0, 1))
            inputs = preprocess(raw_input)

        inputs_batch, targets_batch = tf.train.batch([inputs, targets], batch_size=self.batch_size)
        steps_per_epoch = int(math.ceil(math.ceil(len(input_paths)) / self.batch_size))

        return Examples(
            # paths=paths_batch,
            inputs=inputs_batch,
            targets=targets_batch,
            count=len(input_paths),
            steps_per_epoch=steps_per_epoch,
        )

    def l2_norm(input_x, epsilon=1e-12):
        input_x_norm = input_x / (tf.reduce_sum(input_x ** 2) ** 0.5 + epsilon)
        return input_x_norm

    def weights_spectral_norm(w, iteration=12):
        w_shape = w.shape.as_list()
        w = tf.reshape(w, [-1, w_shape[-1]])
        # print("w:",w.shape)#w: (48, 64)   #w: (1024, 128)   w: (2048, 256)
        u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(),
                            trainable=False)

        u_hat = u
        v_hat = None
        for i in range(iteration):
            """
            power iteration
            Usually iteration = 1 will be enough
            """
            # print("u_hat:",i,u_hat.shape)#u_hat: 0 (1, 64)   u_hat: 0 (1, 128)   u_hat: 0 (1, 256)
            v_ = tf.matmul(u_hat, tf.transpose(w))
            # print("v_",v_.shape)#v_ (1, 48)   #v_ (1, 1024)   v_ (1, 2048)
            v_hat = l2_norm(v_)
            # print("v_hat:",v_hat.shape)#v_hat: (1, 48)  v_hat: (1, 1024)   v_hat: (1, 2048)
            u_ = tf.matmul(v_hat, w)
            u_hat = l2_norm(u_)
        sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
        # print("sigma",sigma.shape)#sigma (1, 1)
        w_norm = w / sigma

        with tf.control_dependencies([u.assign(u_hat)]):
            w_norm = tf.reshape(w_norm, w_shape)
        return w_norm

    def conv_D(batch_input, out_channels, stride, padding=1, size=3, update_collection=None):
        with tf.variable_scope("conv"):
            in_channels = batch_input.get_shape()[3]
            filter = tf.get_variable("filter", [size, size, in_channels, out_channels], dtype=tf.float32,
                                     initializer=tf.random_normal_initializer(0, 0.02))
            filter = weights_spectral_norm(filter)
            padded_input = tf.pad(batch_input, [[0, 0], [padding, padding], [padding, padding], [0, 0]],
                                  # padding=0
                                  mode="CONSTANT")
            conv = tf.nn.conv2d(padded_input, filter, [1, stride, stride, 1], padding="VALID")
            return conv

    def conv(batch_input, out_channels, stride, padding=1, size=3):
        with tf.variable_scope("conv"):
            in_channels = batch_input.get_shape()[3]
            filter = tf.get_variable("filter", [size, size, in_channels, out_channels], dtype=tf.float32,
                                     initializer=tf.random_normal_initializer(0, 0.02))  # （0，1）
            # [batch, in_height, in_width, in_channels], [filter_width, filter_height, in_channels, out_channels]
            #     => [batch, out_height, out_width, out_channels]
            padded_input = tf.pad(batch_input, [[0, 0], [padding, padding], [padding, padding], [0, 0]],
                                  # padding=0
                                  mode="CONSTANT")
            conv = tf.nn.conv2d(padded_input, filter, [1, stride, stride, 1], padding="VALID")
            return conv

    def lrelu(x, a):
        with tf.name_scope("lrelu"):
            # adding these together creates the leak part and linear part
            # then cancels them out by subtracting/adding an absolute value term
            # leak: a*x/2 - a*abs(x)/2
            # linear: x/2 + abs(x)/2

            # this block looks like it has 2 inputs on the graph unless we do this
            x = tf.identity(x)
            return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)

    def batchnorm(input):
        with tf.variable_scope("batchnorm"):
            # this block looks like it has 3 inputs on the graph unless we do this
            input = tf.identity(input)

            channels = input.get_shape()[3]
            offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
            scale = tf.get_variable("scale", [channels], dtype=tf.float32,
                                    initializer=tf.random_normal_initializer(0, 0.02))
            mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
            variance_epsilon = 1e-5
            normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale,
                                                   variance_epsilon=variance_epsilon)
            return normalized

    def deconv(batch_input, out_channels):
        with tf.variable_scope("deconv"):
            batch, in_height, in_width, in_channels = [int(d) for d in batch_input.get_shape()]
            filter = tf.get_variable("filter", [3, 3, out_channels, in_channels], dtype=tf.float32,
                                     initializer=tf.random_normal_initializer(0, 0.02))
            # [batch, in_height, in_width, in_channels], [filter_width, filter_height, out_channels, in_channels]
            #     => [batch, out_height, out_width, out_channels]
            conv = tf.nn.conv2d_transpose(batch_input, filter,
                                          [batch, in_height * 2, in_width * 2, out_channels],
                                          [1, 2, 2, 1], padding="SAME")
            return conv
    def create_generator(generator_inputs, generator_outputs_channels):
        layers = []
        with tf.variable_scope("encoder_1"):
            output = conv(generator_inputs, self.ngf, stride=2, padding=1)
            layers.append(output)

        layer_specs = [
            self.ngf * 2,
            self.ngf * 4,
            self.ngf * 8,
        ]

        for out_channels in layer_specs:
            with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
                rectified = lrelu(layers[-1], 0.2)
                convolved = conv(rectified, out_channels, stride=2, padding=1)
                output = batchnorm(convolved)
                layers.append(output)
        layer_specs = [
            (self.ngf * 4, 0),
            (self.ngf * 2, 0.2),
            (self.ngf * 1, 0.2),
        ]
        num_encoder_layers = len(layers)
        for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
            skip_layer = num_encoder_layers - decoder_layer - 1
            with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
                if decoder_layer == 0:
                    input = tf.random_normal(shape=layers[-1].shape)
                else:
                    input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

                rectified = lrelu(input, 0.2)
                output = deconv(rectified, out_channels)
                output = batchnorm(output)
                if dropout > 0.0:
                    output = tf.nn.dropout(output, keep_prob=1 - dropout)
                layers.append(output)
        with tf.variable_scope("decoder_1"):
            input = tf.concat([layers[-1], layers[0]], axis=3)
            # # rectified = tf.nn.relu(input)
            rectified = deconv(input, generator_outputs_channels)
            output = tf.tanh(rectified)
            layers.append(output)

        return layers[-1]

    def create_model(inputs, targets):
        def create_discriminator(discrim_inputs, discrim_targets, update_collection):
            n_layers = 1
            layers = []
            with tf.variable_scope("concat_c_t"):
                input = tf.concat([discrim_inputs, discrim_targets], axis=3)
            with tf.variable_scope("layer_1"):
                convolved = conv_D(input, self.ndf, stride=2, padding=1, size=3,
                                   update_collection=update_collection)
                convolved = batchnorm(convolved)
                rectified = lrelu(convolved, 0.2)
                layers.append(rectified)
            stride = 2
            padding = 1
            for i in range(n_layers):
                with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                    out_channels = self.ndf * min(2 ** (i + 1), 8)
                    convolved = conv_D(layers[-1], out_channels, stride=stride, padding=padding, size=3,
                                       update_collection=update_collection)
                    normalized = batchnorm(convolved)
                    rectified = lrelu(normalized, 0.2)
                    layers.append(rectified)
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                convolved = conv_D(rectified, out_channels=1, stride=2, padding=0, size=2,
                                   update_collection=update_collection)
                convolved = batchnorm(convolved)
                output = tf.sigmoid(convolved)
                layers.append(output)

            return layers[-1]

        with tf.variable_scope("generator") as scope:
            out_channels = int(targets.get_shape()[-1])
            outputs = create_generator(inputs, out_channels)

        with tf.name_scope("real_discriminator"):
            with tf.variable_scope("discriminator"):
                predict_real = create_discriminator(inputs, targets, update_collection=None)

        with tf.name_scope("fake_discriminator"):
            with tf.variable_scope("discriminator", reuse=True):
                predict_fake = create_discriminator(inputs, outputs, update_collection='NO_OPS')
        return Model(
            outputs=outputs,
        )

    if self.seed is None:
        self.seed = random.randint(0, 2 ** 31 - 1)

    tf.set_random_seed(self.seed)
    np.random.seed(self.seed)
    random.seed(self.seed)
    if not os.path.exists(self.output_dir):
        os.makedirs(self.output_dir)

    examples = load_examples()
    model = create_model(examples.inputs, examples.targets)
    outputs = deprocess(model.outputs)
    saver = tf.train.Saver(max_to_keep=1)

    logdir = self.output_dir if (self.freq > 0 or self.freq > 0) else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with sv.managed_session(config=config) as sess:
        if self.checkpoint != '':
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(self.checkpoint)
            saver.restore(sess, checkpoint)
        if self.mode == "test":
            max_steps = examples.steps_per_epoch
            input_paths = glob.glob(os.path.join(self.input_dir, "*.tfrecords"))
            reference_sample = os.listdir(self.variables_path)
            shape = image.open(self.variables_path + '\\' + reference_sample[0])
            for h in range(100):
                join_image = np.zeros(shape=(self.targets, shape.height, shape.width))
                for i in range(max_steps):
                    result_output = sess.run(outputs)
                    sample_index = input_paths[i].split('\\')[-1].replace('.tfrecords', '').split('_')
                    x = int(sample_index[2])
                    y = int(sample_index[3])
                    for j in range(self.targets):
                        result_output_temp = np.reshape(result_output[:, :, :, j:j + 1], (self.size, self.size))
                        join_image[j:j + 1, y:y + self.size, x:x + self.size] = result_output_temp
                        j += 1
                    i += 1
                for k in range(self.targets):
                    join_image[k:k + 1] = join_image[k:k + 1]
                    imwrite(self.output_dir + '\\' + 'Ni_Stochastic_simulation_' + str(h) + '.tif',
                            np.reshape(join_image[k:k + 1], (shape.height, shape.width)))

                    k += 1
                h += 1