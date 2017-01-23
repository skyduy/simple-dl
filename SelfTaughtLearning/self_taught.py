#!/usr/bin/python
# coding: utf-8

"""
    ...
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    Author: YuJun <cuteuy@gmail.com>
"""

import numpy as np
import scipy.io
import scipy.optimize
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from utils import load_mnist_labels, load_mnist_images, display_network
from SparseAutoEncoder.SAE import SparseAutoEncoder
from SoftmaxRegression.SR import SoftmaxRegression


def self_taught():
    """
    此处使用标签大于5的大量数据进行自学习，学到提取特征的模型。
    之后将提取到的模型应用于小于5的数据上，之后再将其训练。

    这里自学习的数据和用来训练/测试的数据标签完全不同，训练的数据少了，但是准确率却提高了。

    这也是自学习的强大之处：先使用非标注的数据提取精华，之后用提取精华的方法先提取，再训练和测试。

    若为先提取自编码，则准确率为96%，此处为98%
    """
    print 'Loading data sets...'
    images = load_mnist_images('data/train-images.idx3-ubyte')
    labels = load_mnist_labels('data/train-labels.idx1-ubyte')

    # 将原始数据划分两份，记为A和B，A用于自学习，B用于训练和测试
    unlabeled_index = np.argwhere(labels >= 5)[:, 0].flatten()
    labeled_index = np.argwhere(labels < 5)[:, 0].flatten()

    # 将B数据再平分为两份，一份用于训练，一份用于测试结果
    num_train = round(labeled_index.shape[0] / 2)
    train_index = labeled_index[0:num_train]
    test_index = labeled_index[num_train:]
    train_data = images[:, train_index]
    train_labels = labels[train_index]
    test_data = images[:, test_index]
    test_labels = labels[test_index]

    # A数据准备进行自学习
    unlabeled_data = images[:, unlabeled_index]

    print '# examples in unlabeled set: {0:d}\n'.format(unlabeled_data.shape[1])
    print '# examples in supervised training set: {0:d}\n'.format(train_data.shape[1])
    print '# examples in supervised testing set: {0:d}\n'.format(test_data.shape[1])

    input_size = 28 * 28
    hidden_size = 196

    # ----自学习，获取特征----------------------
    print 'Sparse Auto Encoder training...'
    sparsity_param = 0.1
    lambda_ = 3e-3
    beta = 3
    encoder = SparseAutoEncoder(input_size, hidden_size, sparsity_param, lambda_, beta)
    opt_solution = scipy.optimize.minimize(
        encoder.loss_value, encoder.theta, args=(unlabeled_data,),
        method='L-BFGS-B', jac=True, options={'maxiter': 400, 'disp': True}
    )
    opt_theta = opt_solution.x
    w1 = opt_theta[0:hidden_size * input_size].reshape(hidden_size, input_size)

    image_w1 = display_network(w1.transpose())
    fig1 = plt.figure()
    plt.imshow(image_w1, cmap=plt.cm.gray)
    plt.title('AutoEncoder Weight.')
    plt.show()

    joblib.dump(opt_theta, 'model/SAE_theta.pkl')

    # ----将训练/测试数据自编码------------------
    print 'Sparse auto encoding the train and test data...'
    the_opt_theta = joblib.load('model/SAE_theta.pkl')
    num_labels = 5
    lambda_ = 1e-4
    train_features = SparseAutoEncoder.feed_forward(the_opt_theta, train_data, input_size, hidden_size)
    test_features = SparseAutoEncoder.feed_forward(the_opt_theta, test_data, input_size, hidden_size)

    # ----使用自编码后的特征进行Softmax训练------
    print 'Softmax Regression training...'
    regressor = SoftmaxRegression(hidden_size, num_labels, lambda_)
    opt_solution = scipy.optimize.minimize(
        regressor.softmax_cost, regressor.theta, args=(train_features, train_labels,),
        method='L-BFGS-B', jac=True, options={'maxiter': 100, 'disp': True}
    )
    opt_theta = opt_solution.x
    joblib.dump(opt_theta, 'model/SR_theta.pkl')

    # ----测试模型准确度-----------------------
    print 'Predicting the data with the trained model...'
    opt_theta = joblib.load('model/SR_theta.pkl')
    predictions = SoftmaxRegression.predict_out(opt_theta, test_features, num_labels, hidden_size)
    correct = test_labels[:, 0] == predictions[:, 0]
    print "Accuracy :", np.mean(correct)


if __name__ == '__main__':
    self_taught()
