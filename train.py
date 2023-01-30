import tensorflow as tf
import os
import json
import glob
import random
import collections
import math
import time
from imageio import imwrite
from os import listdir
import numpy as np

def model_train(self):
    self.variables_path = '/training area/images exported from ArcGIS/'
    self.input_dir = '/training area/tfrecords file/'
    self.output_dir = '/model saved after training/'
    self.mode = 'train'
    self.checkpoint = 'None'  #pre-trained model
    self.max_epochs = 5000  #total iterations
    self.freq = 50  #
    self.conditions = 1  #the number of layers of coarse data
    self.targets = 1   #the number of layers of fine data
    self.l1_weight = 1
    self.gan_weight = 1
    self.size = 16
    self.batch_size = 300 #the batch size (the number of samples entering the model in one iteration) was set to 300 which was experimentally found to stabilize the loss functions
    self.ngf = int((64 / 3) * self.targets)
    self.ndf = int((64 / 6) * (self.targets + self.conditions))

    # EPS= 1e-12
    EPS = 0
    Examples = collections.namedtuple("Examples", "inputs, targets, count, steps_per_epoch")
    Model = collections.namedtuple("Model",
                                   "predict_real,predict_fake,discrim_loss,discrim_grads_and_vars,gen_loss_GAN,gen_loss_L1,gen_loss_cosin,gen_loss,gen_grads_and_vars,outputs,train")
    super_parameter = {'self.freq': self.freq, 'self.conditions': self.conditions, 'self.targets': self.targets,
                       'self.size': self.size, 'self.variables_path': self.variables_path}
    with open(self.output_dir + '\\' + 'super_parameter.txt', mode='w', encoding='utf-8') as f:
        f.write(json.dumps(super_parameter))
        f.close()

    def preprocess(image):
        with tf.name_scope("preprocess"):
            return image * 2 - 1
            # return image

    def deprocess(image):
        with tf.name_scope("deprocess"):
            #         # [-1, 1] => [0, 1]
            return (image + 1) / 2

    def conv(batch_input, out_channels, stride, padding=1, size=3):
        with tf.variable_scope("conv"):
            in_channels = batch_input.get_shape()[3]
            filter = tf.get_variable("filter", [size, size, in_channels, out_channels], dtype=tf.float32,
                                     initializer=tf.random_normal_initializer(0, 0.02))
            padded_input = tf.pad(batch_input, [[0, 0], [padding, padding], [padding, padding], [0, 0]],
                                  # padding=0
                                  mode="CONSTANT")
            conv = tf.nn.conv2d(padded_input, filter, [1, stride, stride, 1], padding="VALID")
            return conv

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

    def l2_norm(input_x, epsilon=1e-12):
        input_x_norm = input_x / (tf.reduce_sum(input_x ** 2) ** 0.5 + epsilon)
        return input_x_norm

    def weights_spectral_norm(w, iteration=12):
        w_shape = w.shape.as_list()
        w = tf.reshape(w, [-1, w_shape[-1]])
        # print("w:",w.shape)#w: (48, 64)   #w: (1024, 128)   w: (2048, 256)
        u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

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

    def lrelu(x, a):
        with tf.name_scope("lrelu"):
            x = tf.identity(x)
            return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)

    def batchnorm(input):
        with tf.variable_scope("batchnorm"):
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
            conv = tf.nn.conv2d_transpose(batch_input, filter,
                                          [batch, in_height * 2, in_width * 2, out_channels],
                                          [1, 2, 2, 1], padding="SAME")
            return conv

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
            raw_input = tf.reshape(raw_input, [self.size, self.size, self.conditions + self.targets])
            raw_input = tf.identity(raw_input)
            targets = preprocess(raw_input[:, :, :self.targets])
            inputs = preprocess(raw_input[:, :, self.targets:])
        inputs_batch, targets_batch = tf.train.batch([inputs, targets], batch_size=self.batch_size)
        steps_per_epoch = int(math.ceil(math.ceil(len(input_paths)) / self.batch_size))

        return Examples(
            inputs=inputs_batch,
            targets=targets_batch,
            count=len(input_paths),
            steps_per_epoch=steps_per_epoch,
        )

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

        with tf.name_scope("discriminator_loss"):
            discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))
        with tf.name_scope("generator_loss"):
            gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
            gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
            x1_norm = tf.sqrt(tf.reduce_sum(tf.square(targets), axis=2))
            x2_norm = tf.sqrt(tf.reduce_sum(tf.square(outputs), axis=2))
            x1_x2 = tf.reduce_sum(tf.multiply(targets, outputs), axis=2)
            gen_loss_cosin = -tf.reduce_mean(tf.divide(x1_x2, tf.multiply(x1_norm, x2_norm)))
            gen_loss = gen_loss_GAN * self.gan_weight + gen_loss_L1 + gen_loss_cosin
        with tf.name_scope("discriminator_train"):
            discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
            discrim_optim = tf.train.AdamOptimizer()
            discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
            discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

        with tf.name_scope("generator_train"):
            with tf.control_dependencies([discrim_train]):
                gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
                gen_optim = tf.train.AdamOptimizer()
                gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
                gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

        ema = tf.train.ExponentialMovingAverage(decay=0.99)
        update_losses = ema.apply([gen_loss, discrim_loss, gen_loss_GAN, gen_loss_L1, gen_loss_cosin])

        global_step = tf.contrib.framework.get_or_create_global_step()
        incr_global_step = tf.assign(global_step, global_step + 1)
        return Model(
            predict_real=predict_real,
            predict_fake=predict_fake,
            discrim_loss=ema.average(discrim_loss),
            discrim_grads_and_vars=discrim_grads_and_vars,
            gen_loss_GAN=ema.average(gen_loss_GAN),
            gen_loss_L1=ema.average(gen_loss_L1),
            gen_loss_cosin=ema.average(gen_loss_cosin),
            gen_loss=ema.average(gen_loss),
            gen_grads_and_vars=gen_grads_and_vars,
            outputs=outputs,
            train=tf.group(update_losses, incr_global_step, gen_train),
        )

    if not os.path.exists(self.output_dir):
        os.makedirs(self.output_dir)
    examples = load_examples()
    print("examples count = %d" % examples.count)
    model = create_model(examples.inputs, examples.targets)

    inputs = deprocess(examples.inputs)
    targets = deprocess(examples.targets)
    outputs = deprocess(model.outputs)

    # summaries
    with tf.name_scope("inputs_summary"):
        conditions_list = listdir(self.variables_path)[self.targets:]
        i = 0
        for ele in conditions_list:
            tf.summary.image("inputs_{0}".format(ele), inputs[:, :, :, i:i + 1])
            i += 1
    with tf.name_scope("targets_summary"):
        targets_list = listdir(self.variables_path)[:self.targets]
        i = 0
        for ele in targets_list:
            tf.summary.image("target_{0}".format(ele), targets[:, :, :, i:i + 1])
            i += 1
    with tf.name_scope("outputs_summary"):
        targets_output = listdir(self.variables_path)[:self.targets]
        i = 0
        for ele in targets_output:
            tf.summary.image("output_{0}".format(ele), outputs[:, :, :, i:i + 1])
            i += 1

    with tf.name_scope("predict_real_summary"):
        tf.summary.image("predict_real", tf.image.convert_image_dtype(model.predict_real, dtype=tf.float32))

    with tf.name_scope("predict_fake_summary"):
        tf.summary.image("predict_fake", tf.image.convert_image_dtype(model.predict_fake, dtype=tf.float32))

    tf.summary.scalar("discriminator_loss", model.discrim_loss)
    tf.summary.scalar("generator_loss_GAN", model.gen_loss_GAN)
    tf.summary.scalar("generator_loss_L1", model.gen_loss_L1)
    tf.summary.scalar("generator_loss_cosin", model.gen_loss_cosin)
    tf.summary.scalar("generator_loss", model.gen_loss)
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + "/values", var)

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
        max_steps = examples.steps_per_epoch * self.max_epochs
        start = time.time()
        for step in range(max_steps):
            def should(freq):
                return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

            options = None
            run_metadata = None
            if should(self.freq):
                options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

            fetches = {
                "train": model.train,
                "global_step": sv.global_step,
            }

            if should(self.freq):
                fetches["discrim_loss"] = model.discrim_loss
                fetches["gen_loss_GAN"] = model.gen_loss_GAN
                fetches["gen_loss_L1"] = model.gen_loss_L1
                fetches["gen_loss_cosin"] = model.gen_loss_cosin
                fetches["gen_loss"] = model.gen_loss
            if should(self.freq):
                fetches["summary"] = sv.summary_op

            results = sess.run(fetches, options=options, run_metadata=run_metadata)
            if should(self.freq):
                print("recording summary")
                sv.summary_writer.add_summary(results["summary"], results["global_step"])

            if should(self.freq):
                print("recording trace")
                sv.summary_writer.add_run_metadata(run_metadata, "step_%d" % results["global_step"])

            if should(self.freq):
                train_epoch = math.ceil(results["global_step"] / examples.steps_per_epoch)
                train_step = (results["global_step"] - 1) % examples.steps_per_epoch + 1
                rate = (step + 1) * self.batch_size / (time.time() - start)
                remaining = (max_steps - step) * self.batch_size / rate
                print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (
                    train_epoch, train_step, rate, remaining / 60))
                print("discrim_loss", results["discrim_loss"])
                print("gen_loss_GAN", results["gen_loss_GAN"])
                print("gen_loss_L1", results["gen_loss_L1"])
                print("gen_loss_cosin", results["gen_loss_cosin"])
                print("gen_loss", results["gen_loss"])
            if should(self.freq):
                print("saving model")
                saver.save(sess, os.path.join(self.output_dir, "model"), global_step=sv.global_step)
            if sv.should_stop():
                    break