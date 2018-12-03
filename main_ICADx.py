import tensorflow as tf
import numpy as np
import random
import time
import scipy.io as sio
import model_vgg

'''
##########################################################################################
                DB
##########################################################################################
'''
import h5py


def load_data():
    DDSM = h5py.File('./data/DDSM.h5', 'r')

    X_train = DDSM['X_train']
    X_test = DDSM['X_test']
    cad_train = DDSM['cad_train']
    cad_test = DDSM['cad_test']
    margins_train = DDSM['margins_train']
    margins_test = DDSM['margins_test']
    shape_train = DDSM['shape_train']
    shape_test = DDSM['shape_test']

    return (
    X_train, cad_train, margins_train, shape_train, X_test, cad_test, margins_test, shape_test)


for fold in range(1):
    X_train, cad_train, margin_train, shape_rain, X_test, cad_test, margin_test, shape_test = load_data()

    '''
    ##########################################################################################
                    model
    ##########################################################################################
    '''

    X = tf.placeholder("float", [None, 64, 64, 3])
    y_train0 = tf.placeholder("int32", [None, len(cad_train[0])])
    y_train1 = tf.placeholder("int32", [None, len(margin_train[0])])
    y_train2 = tf.placeholder("int32", [None, len(shape_rain[0])])

    learning_rate = tf.placeholder("float", [])

    VGG = model_vgg.vgg16base(X)

    cross_entropy0 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=[VGG.cad], labels=y_train0))
    cross_entropy1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=[VGG.margins], labels=y_train1))
    cross_entropy2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=[VGG.shape], labels=y_train2))
    cross_entropy = cross_entropy0 + 0.5 * cross_entropy1 + 0.5 * cross_entropy2

    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.1
    lamda = 0.0001

    bta1 = 0.9
    bta2 = 0.999
    epsln = 0.00001

    train_op = tf.train.AdamOptimizer(learning_rate, 0.9).minimize(cross_entropy, global_step=global_step)
    # obj_CPerformanceMonitor = CPerformanceMonitor()
    auc_var = tf.contrib.metrics.streaming_auc(tf.nn.softmax(VGG.cad), y_train0, num_thresholds=10000)

    batch_size = 16
    epoch = 100

    with tf.Session() as sess:

        basemodel_path = "./model"
        LegendAdded = False
        trainloss = []
        trainAcc = []
        testloss = []
        testAcc = []
        arr_iter = []

        sess.run(tf.global_variables_initializer())
        correct_prediction0 = tf.equal(tf.argmax(VGG.cad, 1), tf.argmax(y_train0, 1))
        accuracy0 = tf.reduce_mean(tf.cast(correct_prediction0, "float"))
        correct_prediction1 = tf.equal(tf.argmax(VGG.margins, 1), tf.argmax(y_train1, 1))
        accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, "float"))
        correct_prediction2 = tf.equal(tf.argmax(VGG.shape, 1), tf.argmax(y_train2, 1))
        accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, "float"))

        VGG.load_weights('vgg16_weights_nonfc.npz', sess)

        saver = tf.train.Saver()


        for j in range(epoch):
            start_time = time.time()
            for i in range(0, len(X_train), batch_size):
                if i + batch_size < len(X_train):
                    lr = 0.000001
                    if j > 180:
                        lr = 0.000001
                    if j > 195:
                        lr = 0.0000001

                    if j > -1:
                        batch_cifar_aug = np.zeros([batch_size, 64, 64, 3])
                        for l in range(i, i + batch_size):
                            image = X_train[l]

                            # rand_R = random.randrange(0, 4)
                            # image = np.rot90(image, rand_R)

                            npad = ((2, 2), (2, 2), (0, 0))
                            image = np.pad(image, pad_width=npad, mode='constant', constant_values=0)

                            rand_x = random.randrange(0, 4)
                            rand_y = random.randrange(0, 4)

                            image = image[rand_x:rand_x + 64, rand_y:rand_y + 64]

                            filp1 = random.randrange(0, 2)
                            if filp1 == 0:
                                image = np.flip(image, 1)
                            batch_cifar_aug[l - i] = image

                        feed_dict = {
                            X: batch_cifar_aug,
                            y_train0: cad_train[i:i + batch_size],
                            y_train1: margin_train[i:i + batch_size],
                            y_train2: shape_rain[i:i + batch_size],
                            learning_rate: lr}

                        _ = sess.run([train_op], feed_dict=feed_dict)
                        print "training on image #%d" % i
                    else:
                        feed_dict = {
                            X: X_train[i:i + batch_size],
                            y_train0: cad_train[i:i + batch_size],
                            y_train1: margin_train[i:i + batch_size],
                            y_train2: shape_rain[i:i + batch_size],
                            learning_rate: lr}
                        _ = sess.run([train_op], feed_dict=feed_dict)
                        print "training on image #%d" % i

                else:
                    feed_dict = {
                        X: X_train[i:],
                        y_train0: cad_train[i:i + batch_size],
                        y_train1: margin_train[i:i + batch_size],
                        y_train2: shape_rain[i:i + batch_size],
                        learning_rate: lr}
                    _ = sess.run([train_op], feed_dict=feed_dict)

                if i % 1280 == 0:
                    print "epoch %d" % j
                    print "training on image #%d" % i
                    loss_train0 = sess.run([cross_entropy],
                                           feed_dict={
                                               X: X_train[0:100],
                                               y_train0: cad_train[0:100],
                                               y_train1: margin_train[0:100],
                                               y_train2: shape_rain[0:100]})
                    print("Train loss= " + "{:.6f}".format(loss_train0[0]))

                    acc_train0 = sess.run([accuracy0],
                                          feed_dict={
                                              X: X_test,
                                              y_train0: cad_test})
                    print("Test Acc0= " + "{:.6f}".format(acc_train0[0]))

                    acc_train1 = sess.run([accuracy1],
                                          feed_dict={
                                              X: X_test,
                                              y_train1: margin_test})
                    print("Test Acc1= " + "{:.6f}".format(acc_train1[0]))
                    acc_train2 = sess.run([accuracy2],
                                          feed_dict={
                                              X: X_test,
                                              y_train2: shape_test})
                    print("Test Acc2= " + "{:.6f}".format(acc_train2[0]))

                    testAcc.append(acc_train0)


            print("epoch Time: %s seconds *****" % (time.time() - start_time))

            cad = sess.run(tf.nn.softmax(VGG.cad), feed_dict={
                X: X_test,
                y_train0: cad_test})
            margins = sess.run(tf.nn.softmax(VGG.margins), feed_dict={
                X: X_test,
                y_train1: margin_test})
            shape = sess.run(tf.nn.softmax(VGG.shape), feed_dict={
                X: X_test,
                y_train2: shape_test})

            savepath = basemodel_path + '/%d epoch %dfold' % (j, fold)
            sio.savemat(savepath, mdict={'cad': cad,
                                         'margins': margins,
                                         'shapes': shape
                                         })

            saver.save(sess, basemodel_path + '/model.ckpt')


        sess.close()
