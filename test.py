from __future__ import division, print_function
import time
import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.python.platform import gfile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import utils
import unet


NX = 512
NY = 512

checkpoint_dir = './check_point/'

generator = utils.ImageGenerator(NX, NY, cnt=20)

with tf.Session() as sess:
    in_image = tf.placeholder(tf.float32, [None, None, None, generator.channels])
    label_image = tf.placeholder(tf.float32, [None, None, None, generator.n_class])
    out_image = unet.network(in_image)
 
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt:
        print('loaded ' + ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print('Error: checkpoint not exists')
        exit(1)

    input_x, input_y = generator(1)

    output = sess.run(out_image, feed_dict={in_image: input_x, label_image: input_y})

    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12,5))
    ax[0].imshow(input_x[0,...,0], aspect="auto")
    ax[1].imshow(input_y[0,...,1], aspect="auto")
    ax[2].imshow(output[0,...,1] > 0.9, aspect="auto")
    ax[0].set_title("Input")
    ax[1].set_title("Ground truth")
    ax[2].set_title("Prediction")
    fig.tight_layout()
    fig.savefig("./result/result.png")