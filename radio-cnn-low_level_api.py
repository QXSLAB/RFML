"""Convolutional Neural Network Estimator for RadioML, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import radio_data
import os

# tf.logging.set_verbosity(tf.logging.INFO)

step_display = 1
path = '/Users/ty/Desktop/models'
model_sn = 1000
batch_size = 500
snr = 10


def main(_):

    with tf.name_scope('database'):
        # Import feature and label
        train_features, train_onehot = radio_data.read_train_data(snr)
        test_features, test_onehot = radio_data.read_test_data(snr)
        assert train_features.shape[0] == train_onehot.shape[0]
        assert test_features.shape[0] == test_onehot.shape[0]
        # Define database
        train_features_placeholder = tf.placeholder(train_features.dtype, train_features.shape)
        train_onehot_placeholder = tf.placeholder(train_onehot.dtype, train_onehot.shape)
        train_dataset = tf.contrib.data.Dataset.from_tensor_slices((train_features_placeholder, train_onehot_placeholder))
        train_dataset = train_dataset.shuffle(buffer_size=500)
        train_dataset = train_dataset.batch(batch_size)
        train_iterator = train_dataset.make_initializable_iterator()
        next_element = train_iterator.get_next()

    # Dropout rate
    dr = tf.placeholder(tf.float32)

    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # Radio Data are 2x128 points, and have 1 channels
    output1 = tf.placeholder(tf.float32, [None, 2, 128, 1], name="input")
    onehot_labels = tf.placeholder(tf.float32, [None, radio_data.class_num], name="GT")

    # Convolutional Layer
    # Computes 256 features using a 1x3 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 2, 128, 1]
    # Output Tensor Shape: [batch_size, 2, 128, 256]
    weight2 = tf.Variable(tf.truncated_normal([1, 3, 1, 256], stddev=.1))
    bias2 = tf.Variable(tf.constant(0.1, shape=[256]))
    conv2 = tf.nn.conv2d(input=output1, filter=weight2, strides=[1, 1, 1, 1], padding="SAME", name="conv")
    wx_plus_b2 = conv2+bias2
    axis2 = list(range(len(wx_plus_b2.shape) - 1))
    wb_mean2, wb_var2 = tf.nn.moments(wx_plus_b2, axis2)
    scale2 = tf.Variable(tf.ones(wb_mean2.shape))
    offset2 = tf.Variable(tf.zeros(wb_mean2.shape))
    wx_plus_b_n2 = tf.nn.batch_normalization(wx_plus_b2, wb_mean2, wb_var2, offset2, scale2, 0.001)
    act2 = tf.nn.relu(wx_plus_b_n2, name="relu")
    output2 = tf.nn.dropout(act2, keep_prob=1)

    # Convolutional Layer
    # Computes 80 features using a 2x3 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 2, 128, 256]
    # Output Tensor Shape: [batch_size, 2, 128, 80]
    weight3 = tf.Variable(tf.truncated_normal([2, 3, 256, 80], stddev=.1))
    bias3 = tf.Variable(tf.constant(0.1, shape=[80]))
    conv3 = tf.nn.conv2d(input=output2, filter=weight3, strides=[1, 1, 1, 1], padding="SAME", name="conv")
    wx_plus_b3 = conv3+bias3
    axis3 = list(range(len(wx_plus_b3.shape) - 1))
    wb_mean3, wb_var3 = tf.nn.moments(wx_plus_b3, axis3)
    scale3 = tf.Variable(tf.ones(wb_mean3.shape))
    offset3 = tf.Variable(tf.zeros(wb_mean3.shape))
    wx_plus_b_n3 = tf.nn.batch_normalization(wx_plus_b3, wb_mean3, wb_var3, offset3, scale3, 0.001)
    act3 = tf.nn.relu(wx_plus_b_n3, name="relu")
    dropout3 = tf.nn.dropout(act3, keep_prob=1)
    output3 = tf.reshape(dropout3, [-1, 2*128*80])

    # Dense Layer
    # Densely connected layer with 256 neurons
    # Input Tensor Shape: [batch_size, 128*80]
    # Output Tensor Shape: [batch_size, 256]
    weight4 = tf.Variable(tf.truncated_normal([2*128*80, 256], stddev=0.1))
    bias4 = tf.Variable(tf.constant(0.1, shape=[256]))
    wx_plus_b4 = tf.matmul(output3, weight4) + bias4
    axis4 = list(range(len(wx_plus_b4.shape) - 1))
    wb_mean4, wb_var4 = tf.nn.moments(wx_plus_b4, axis4)
    scale4 = tf.Variable(tf.ones(wb_mean4.shape))
    offset4 = tf.Variable(tf.zeros(wb_mean4.shape))
    wx_plus_b_n4 = tf.nn.batch_normalization(wx_plus_b4, wb_mean4, wb_var4, offset4, scale4, 0.001)
    act4 = tf.nn.relu(wx_plus_b_n4)
    output4 = tf.nn.dropout(act4, keep_prob=dr)

    # Logits layer
    # Input Tensor Shape: [batch_size, 256]
    # Output Tensor Shape: [batch_size, 10]
    weight5 = tf.Variable(tf.truncated_normal([256, 10], stddev=.1))
    bias5 = tf.Variable(tf.constant(0.1, shape=[10]))
    logits = tf.nn.relu(tf.matmul(output4, weight5) + bias5)

    # Calculate Loss
    loss_op = tf.losses.softmax_cross_entropy(onehot_labels=tf.cast(onehot_labels, tf.int32), logits=logits)
    loss_summ = tf.summary.scalar('loss', loss_op)

    global_step = tf.Variable(0, trainable=False)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(loss=loss_op, global_step=global_step)
    w5_summ = tf.summary.histogram('w5', weight5)
    b5_summ = tf.summary.histogram('b5', bias5)
    w4_summ = tf.summary.histogram('w4', weight4)
    b4_summ = tf.summary.histogram('b4', bias4)
    w5_g_summ = tf.summary.histogram('w5_g', optimizer.compute_gradients(loss_op, var_list=[weight5]))
    b5_g_summ = tf.summary.histogram('b5_g', optimizer.compute_gradients(loss_op, var_list=[bias5]))
    w4_g_summ = tf.summary.histogram('w4_g', optimizer.compute_gradients(loss_op, var_list=[weight4]))
    b4_g_summ = tf.summary.histogram('b4_g', optimizer.compute_gradients(loss_op, var_list=[bias4]))

    # Calculate accuracy
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(onehot_labels, 1))
    train_acc_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    test_acc_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    test_summ = tf.summary.scalar('test', test_acc_op)
    train_acc_summ = tf.summary.scalar('train', train_acc_op)

    train_summ = tf.summary.merge([loss_summ, train_acc_summ,
                                   w5_summ, b5_summ, w4_summ, b4_summ, w5_g_summ, b5_g_summ, w4_g_summ, b4_g_summ])

    with tf.Session() as sess:

        model_paras = "model.ckpt"

        train_writer = tf.summary.FileWriter(path, sess.graph)
        saver = tf.train.Saver()

        if os.path.exists(path+"/"+"checkpoint"):
            saver.restore(sess, path+"/"+model_paras+model_sn)
            print('model restored')
        else:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)

        # Train
        for epoch in range(20000):
            sess.run(train_iterator.initializer, feed_dict={train_features_placeholder: train_features,
                                                            train_onehot_placeholder: train_onehot})
            while True:
                try:
                    batch = sess.run(next_element)
                    sess.run(train_op, feed_dict={output1: batch[0], onehot_labels: batch[1], dr: 0.6})
                    step = global_step.eval()
                    if step % step_display == 0:
                        train_log, loss, train_acc = sess.run(
                            [train_summ, loss_op, train_acc_op],
                            feed_dict={output1: batch[0], onehot_labels: batch[1], dr: 1})
                        test_log, test_acc = sess.run(
                            [test_summ, test_acc_op],
                            feed_dict={output1: test_features, onehot_labels: test_onehot, dr: 1})
                        train_writer.add_summary(train_log, step)
                        train_writer.add_summary(test_log, step)
                        train_writer.flush()
                        print("Epoch: %d, Global Step: %d, loss: %f, train_accuracy: %f, test_accuracy: %f"
                              % (epoch, step, loss, train_acc, test_acc))
                    if step % 1000 == 0:
                        saver.save(sess, path+"/"+model_paras, global_step=step)
                        print("model saved")
                except tf.errors.OutOfRangeError:
                    break
        train_writer.close()


if __name__ == '__main__':
    tf.app.run(main=main)
