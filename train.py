from __future__ import division, print_function
import os, time, scipy.io
import tensorflow as tf
from tensorflow.python.framework import graph_util
import numpy as np
import utils
import unet


NX = 512
NY = 512
LR = 1e-3
DECAY = 0.9
EPOCHS = 100
ITERS = 100

checkpoint_dir = './check_point/'
model_dir = "./model/"

generator = utils.ImageGenerator(NX, NY)
losses = utils.AverageMeter()

with tf.Session() as sess:
    in_image = tf.placeholder(tf.float32, [None, None, None, generator.channels])
    label_image = tf.placeholder(tf.float32, [None, None, None, generator.n_class])
    out_image = unet.network(in_image)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=out_image,
                                                                        labels=label_image))
    optim = tf.train.RMSPropOptimizer(learning_rate=LR, decay=DECAY).minimize(loss)
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt:
        print('loaded ' + ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)

    for epoch in range(EPOCHS):
        input_x, input_y = generator(1)
        losses.reset()

        for iteration in range(ITERS):
            st = time.time()
            _, loss_current, output = sess.run([optim, loss, out_image],
                                                feed_dict={in_image: input_x, label_image: input_y})
            losses.update(loss_current)
            print("[%d][%d]:    Loss=%.3f   Time=%.3f" % (epoch, iteration, losses.avg, time.time() - st))

        saver.save(sess, checkpoint_dir + 'model.ckpt')
