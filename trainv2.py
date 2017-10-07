import os
from scipy.misc import imread, imresize, imsave
import tensorflow as tf
import next_batch
import numpy
import time

def load_file_names(folder_name):
    filenames = []
    for root, dirs, files in os.walk(folder_name, followlinks=True):

        for file in files:
            file_name = os.path.join(root, file)
            direction = file.split(".")[-2]
            filenames.append([file_name, direction])
    print("%d images to be loaded" %len(filenames))
    return numpy.array(filenames)


def load_images(file_names):
    images = numpy.array([])
    labels = numpy.array([])
    for file_name in file_names:
        image = imread(file_name[0], flatten=True)
        resize_image = imresize(image, (128, 128))[64:128, 0:128].flatten()
       # imsave("image1.png", resize_image)

        # single gray scale layer. to do use rgb components
        # print("file_name = %s, file=%s, direction=%s"%(file_name, file, direction))


        images = numpy.concatenate((images, resize_image), axis=0)
        labels = numpy.concatenate((labels, whichdirection(file_name[1])), axis=0)
    images = numpy.reshape(images, [-1, 16384 / 2])
    labels = numpy.reshape(labels, [-1, 4])
    return images, labels


def whichdirection(direction):
    if direction == 'f':
        return [1, 0, 0, 0]
    elif direction == 'l':
        return [0, 1, 0, 0]
    elif direction == 'b':
        return [0, 0, 1, 0]
    elif direction == 'r':
        return [0, 0, 0, 1]


def deepnn(x):
    x_image = tf.reshape(x, [-1, 128,  64, 1])
    w_conv1 = weight_variable([3, 3, 1, 16], v_name="w_conv1")
    b_conv1 = bias_variable([16], v_name="b_conv1")
    h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1)+ b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    w_conv2 = weight_variable([3, 3, 16, 32], v_name="w_conv2")
    b_conv2 = bias_variable([32], v_name="b_conv2")
    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)

    #Second pooling layer
    h_pool2 = max_pool_2x2(h_conv2)

    w_conv3 = weight_variable([3, 3, 32, 32], v_name="w_conv3")
    b_conv3 = bias_variable([32], v_name="b_conv3")
    h_conv3 = tf.nn.relu(conv2d(h_pool2, w_conv3) + b_conv3)

    # third pooling layer
    h_pool3 = max_pool_2x2(h_conv3)

    #fully connectd layer 1 -- after 2 round of downsamplin, our 224x224 image is down to xxx featuere maps -- maps this to 1024 features
    w_fc1 = weight_variable([16 * 8 * 32, 64], v_name="w_fc1")
    b_fc1 = bias_variable([64], v_name="b_fc1")

    hpool3_flat = tf.reshape(h_pool3, [-1, 16 * 8 *32], name="flatten_conv3")
    h_fc1_1 = tf.matmul(hpool3_flat, w_fc1) + b_fc1
    is_training_mode = tf.placeholder(tf.bool, name="is_training")
    h_fc1_bn = tf.contrib.layers.batch_norm(h_fc1_1, center=True, scale=True, is_training=is_training_mode, scope='bn')
    h_fc1 = tf.nn.relu(h_fc1_bn)

    #Dropout - controls the complexity of the model, prevents co-adaptation of features.
    keep_prob = tf.placeholder(tf.float32, name = "dropout_keep_prob")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([64, 4], v_name="w_fc2")
    b_fc2 = bias_variable([4], v_name="b_fc2")
    y_conv = tf.nn.xw_plus_b(h_fc1_drop, W_fc2, b_fc2, name="score")
        #tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv, keep_prob, is_training_mode


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape, v_name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=v_name)

def bias_variable(shape, v_name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=v_name)

def main(_):
    files = load_file_names("/home/jacob/Desktop/Test12V1")
    perm0 = numpy.arange(len(files))
    numpy.random.shuffle(perm0)
    shuffled_files = files[perm0]
    number_of_samples = int(len(shuffled_files) * 0.8)
    training_files = shuffled_files[0:number_of_samples]
    validation_files = shuffled_files[number_of_samples:-1]

    validation_images, validation_labels = load_images(validation_files)
    print(validation_labels)

    with tf.Session() as sess:

        x = tf.placeholder(tf.float32, [None, 16384 / 2], name="input_x")
        y_ = tf.placeholder(tf.float32, [None, 4], name="input_y")
        global_step = tf.Variable(0, name="global_step", trainable=False)
        y_conv, keep_prob, is_training = deepnn(x)
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
        optimizer = tf.train.AdamOptimizer(0.0001)
        correct_predictoin = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

        accuracy = tf.reduce_mean(tf.cast(correct_predictoin, tf.float32))



        #train_step =optimizer.minimize(cross_entropy)
        grads_and_vars=optimizer.compute_gradients(cross_entropy)
        train_step = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_murged = tf.summary.merge(grad_summaries)
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}/n".format(out_dir))
        loss_summary = tf.summary.scalar("loss", cross_entropy)
        acc_summary = tf.summary.scalar("accuracy", accuracy)

        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_murged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model3")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        #saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)



        saver = tf.train.Saver(max_to_keep=5)

        sess.run(tf.global_variables_initializer())

        print('number of training/validation samples = %d/%s :' % (number_of_samples, len(validation_labels)))

        batches = next_batch.next_batch(training_files, number_of_samples, 100, 30)

        #index = 0
        validationerrorprevious = 0
        for batch in batches:
            x_batch, y_batch = load_images(batch)
            _, step, train_summaries = sess.run([train_step,global_step,train_summary_op],
                                                feed_dict={x: x_batch, y_: y_batch, keep_prob: 0.5, is_training:True})
            train_summary_writer.add_summary(train_summaries, step)
            index = tf.train.global_step(sess, global_step)
            if index %  20 ==0:
                #training_accuracy = accuracy.eval(feed_dict={x: x_batch, y_: y_batch, keep_prob: 1.0})
                training_accuracy,train_summaries = sess.run([accuracy, train_summary_op],
                                                             feed_dict={x: x_batch, y_: y_batch, keep_prob: 1.0, is_training:False})
                validation_accuracy, validation_summaries = sess.run([accuracy, dev_summary_op],
                                                                     feed_dict={x: validation_images, y_: validation_labels,
                                                                                keep_prob: 1.0, is_training:False})
                train_summary_writer.add_summary(train_summaries, index)
                dev_summary_writer.add_summary(validation_summaries, step)
                print('step = %d: training accuracy: %f validation accuracy %f' % (step, training_accuracy, validation_accuracy))
                if validation_accuracy > validationerrorprevious:
                    saver.save(sess, checkpoint_prefix, global_step=index)
                    validationerrorprevious = validation_accuracy

        training_accuracy, train_summaries = sess.run([accuracy, train_summary_op],
                                                      feed_dict={x: x_batch, y_: y_batch, keep_prob: 1.0, is_training:False})
        validation_accuracy, validation_summaries = sess.run([accuracy, dev_summary_op],
                                                             feed_dict={x: validation_images, y_: validation_labels,
                                                                        keep_prob: 1.0, is_training:False})
        train_summary_writer.add_summary(train_summaries, index)
        dev_summary_writer.add_summary(validation_summaries, step)
        print('step = %d: training accuracy: %f validation accuracy %f' % (step, training_accuracy, validation_accuracy))
        if validationerrorprevious < validation_accuracy:
            saver.save(sess, checkpoint_prefix, global_step =index)

if __name__ == "__main__":
    tf.app.run(main=main)


