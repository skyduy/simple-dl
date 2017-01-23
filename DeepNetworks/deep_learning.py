#!/usr/bin/python
# coding: utf-8

"""
    深度学习模型：
        1、对原始数据自编码。
        2、对第一层自编码后的数据再次自编码。
        （1，2综合起来即为栈式自编码：由多层稀疏自编码器组成的神经网络，其前一层自编码器的输出作为其后一层自编码器的输入。）
        3、栈式自编码提取后的特征作为训练特征，并进行SoftMax分类训练。
        4、将各层结合起来并进行微调。

    此处，进行预训练的数据就是带标注的数据。
    若直接softmax分类，即与文件夹SoftmaxRegression一致，准确率为92.6%
    这里微调前准确率为88 %，微调后准确率为98%。
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    Author: YuJun <cuteuy@gmail.com>
"""


import numpy
import scipy.io
import scipy.sparse
import scipy.optimize
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from SparseAutoEncoder.SAE import SparseAutoEncoder
from SoftmaxRegression.SR import SoftmaxRegression
from utils import load_mnist_images, load_mnist_labels, display_network


def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))


def get_ground_truth(labels):
    labels = numpy.array(labels).flatten()
    data = numpy.ones(len(labels))
    indptr = numpy.arange(len(labels)+1)

    ground_truth = scipy.sparse.csr_matrix((data, labels, indptr))
    ground_truth = numpy.transpose(ground_truth.todense())

    return ground_truth


def stack2params(stack):
    params = []
    num_layers = len(stack) / 2
    for i in range(num_layers):
        params = numpy.concatenate((params, numpy.array(stack[i, "W"]).flatten()))
        params = numpy.concatenate((params, numpy.array(stack[i, "b"]).flatten()))

    return params


def params2stack(params, net_config):
    stack = {}
    limit0 = 0

    for i in range(len(net_config)-2):
        limit1 = limit0 + net_config[i] * net_config[i+1]
        limit2 = limit1 + net_config[i+1]

        stack[i, "W"] = params[limit0: limit1].reshape(net_config[i+1], net_config[i])
        stack[i, "b"] = params[limit1: limit2].reshape(net_config[i+1], 1)

        limit0 = limit2

    return stack


def stacked_auto_encoder_cost(theta, net_config, lambda_, data, labels):
    input_size = net_config[-2]
    num_classes = net_config[-1]

    limit0 = 0
    limit1 = num_classes * input_size

    softmax_theta = theta[limit0: limit1].reshape(num_classes, input_size)
    stack = params2stack(theta[limit1:], net_config)

    num_layers = len(stack) / 2

    activation = dict([])

    # 初始层
    activation[0] = data

    # 第二层，第三层：自编码层
    for i in range(num_layers):
        activation[i+1] = sigmoid(numpy.dot(stack[i, "W"], activation[i]) + stack[i, "b"])

    # 第四层：softmax分类结果层
    theta_x = numpy.dot(softmax_theta, activation[num_layers])
    hypothesis = numpy.exp(theta_x)
    probabilities = hypothesis / numpy.sum(hypothesis, axis=0)

    # 计算第四层cost
    ground_truth = get_ground_truth(labels)

    cost_examples = numpy.multiply(ground_truth, numpy.log(probabilities))
    traditional_cost = -(numpy.sum(cost_examples) / data.shape[1])

    theta_squared = numpy.multiply(softmax_theta, softmax_theta)
    weight_decay = 0.5 * lambda_ * numpy.sum(theta_squared)

    cost = traditional_cost + weight_decay

    # 计算第三层至第四层参数梯度
    softmax_theta_grad = -numpy.dot(ground_truth - probabilities, numpy.transpose(activation[num_layers]))
    softmax_theta_grad = softmax_theta_grad / data.shape[1] + lambda_ * softmax_theta

    # 反向传播求第二层——第三层参数梯度 和 第一层——第二层参数梯度
    delta = dict([])
    delta[num_layers] = -numpy.multiply(numpy.dot(numpy.transpose(softmax_theta), ground_truth - probabilities),
                                        numpy.multiply(activation[num_layers], 1 - activation[num_layers]))
    for i in range(num_layers-1):
        index = num_layers - i - 1
        delta[index] = numpy.multiply(numpy.dot(numpy.transpose(stack[index, "W"]), delta[index+1]),
                                      numpy.multiply(activation[index], 1 - activation[index]))

    stack_grad = dict([])
    for i in range(num_layers):
        index = num_layers - i - 1
        stack_grad[index, "W"] = numpy.dot(delta[index+1], numpy.transpose(activation[index])) / data.shape[1]
        stack_grad[index, "b"] = numpy.sum(delta[index+1], axis=1) / data.shape[1]

    # 将各参数梯度连接一起
    params_grad = stack2params(stack_grad)
    theta_grad = numpy.concatenate((numpy.array(softmax_theta_grad).flatten(),
                                    numpy.array(params_grad).flatten()))

    return [cost, theta_grad]


def stacked_auto_encoder_predict(theta, net_config, data):
    input_size = net_config[-2]
    num_classes = net_config[-1]

    limit0 = 0
    limit1 = num_classes * input_size

    softmax_theta = theta[limit0: limit1].reshape(num_classes, input_size)
    stack = params2stack(theta[limit1:], net_config)

    num_layers = len(stack) / 2
    activation = data
    for i in range(num_layers):
        activation = sigmoid(numpy.dot(stack[i, "W"], activation) + stack[i, "b"])

    theta_x = numpy.dot(softmax_theta, activation)
    hypothesis = numpy.exp(theta_x)
    probabilities = hypothesis / numpy.sum(hypothesis, axis=0)

    predictions = numpy.zeros((data.shape[1], 1))
    predictions[:, 0] = numpy.argmax(probabilities, axis=0)

    return predictions


def execute_stacked_auto_encoder():

    visible_size = 784    # size of input vector
    hidden_size1 = 225    # size of hidden layer vector of first auto encoder
    hidden_size2 = 196    # size of hidden layer vector of second auto encoder
    rho = 0.1             # desired average activation of hidden units
    lambda_ = 0.003       # weight decay parameter
    beta = 3              # weight of sparsity penalty term
    max_iterations = 200  # number of optimization iterations
    num_classes = 10      # number of classes

    print 'Loading raw training MNIST data...'
    train_data = load_mnist_images('data/train-images.idx3-ubyte')
    train_labels = load_mnist_labels('data/train-labels.idx1-ubyte')

    print 'Running the first AutoEncoder...'
    encoder1 = SparseAutoEncoder(visible_size, hidden_size1, rho, lambda_, beta)
    opt_solution = scipy.optimize.minimize(
        encoder1.loss_value, encoder1.theta, args=(train_data,),
        method='L-BFGS-B', jac=True, options={'maxiter': max_iterations, 'disp': True}
    )
    sae1_theta = opt_solution.x
    joblib.dump(sae1_theta, 'model/SAE1_theta.pkl')

    print 'Preparing data and running the second AutoEncoder...'
    sae1_theta = joblib.load('model/SAE1_theta.pkl')
    sae1_features = SparseAutoEncoder.feed_forward(sae1_theta, train_data, visible_size, hidden_size1)

    encoder2 = SparseAutoEncoder(hidden_size1, hidden_size2, rho, lambda_, beta)
    opt_solution = scipy.optimize.minimize(
        encoder2.loss_value, encoder2.theta, args=(sae1_features,),
        method='L-BFGS-B', jac=True, options={'maxiter': max_iterations, 'disp': True}
    )
    sae2_theta = opt_solution.x
    joblib.dump(sae2_theta, 'model/SAE2_theta.pkl')

    print 'Preparing data and running the Softmax Regressor...'
    sae2_theta = joblib.load('model/SAE2_theta.pkl')
    sae2_features = SparseAutoEncoder.feed_forward(sae2_theta, sae1_features, hidden_size1, hidden_size2)

    regressor = SoftmaxRegression(hidden_size2, num_classes, lambda_)
    opt_solution = scipy.optimize.minimize(
        regressor.softmax_cost, regressor.theta, args=(sae2_features, train_labels,),
        method='L-BFGS-B', jac=True, options={'maxiter': max_iterations, 'disp': True}
    )
    sr_theta = opt_solution.x
    joblib.dump(sr_theta, 'model/SR_theta.pkl')

    sr_theta = joblib.load('model/SR_theta.pkl')

    # 将两层稀疏编码参数和softmax参数融合为需要微调的总体参数
    stack = dict([])

    encoder1_limit0 = 0
    encoder1_limit1 = hidden_size1 * visible_size
    encoder1_limit2 = 2 * hidden_size1 * visible_size
    encoder1_limit3 = 2 * hidden_size1 * visible_size + hidden_size1
    encoder2_limit0 = 0
    encoder2_limit1 = hidden_size2 * hidden_size1
    encoder2_limit2 = 2 * hidden_size2 * hidden_size1
    encoder2_limit3 = 2 * hidden_size2 * hidden_size1 + hidden_size2

    stack[0, "W"] = sae1_theta[encoder1_limit0: encoder1_limit1].reshape(hidden_size1, visible_size)
    stack[1, "W"] = sae2_theta[encoder2_limit0: encoder2_limit1].reshape(hidden_size2, hidden_size1)
    stack[0, "b"] = sae1_theta[encoder1_limit2: encoder1_limit3].reshape(hidden_size1, 1)
    stack[1, "b"] = sae2_theta[encoder2_limit2: encoder2_limit3].reshape(hidden_size2, 1)

    stack_params = stack2params(stack)
    all_theta = numpy.concatenate((sr_theta.flatten(), stack_params.flatten()))
    joblib.dump(all_theta, 'model/ALL_theta.pkl')
    all_theta = joblib.load('model/ALL_theta.pkl')

    print 'Loading raw testing MNIST data...'
    test_data = load_mnist_images('data/t10k-images.idx3-ubyte')
    test_labels = load_mnist_labels('data/t10k-labels.idx1-ubyte')
    net_config = [visible_size, hidden_size1, hidden_size2, num_classes]

    print 'Predicting before fine tuning...'
    predictions = stacked_auto_encoder_predict(all_theta, net_config, test_data)
    correct = test_labels[:, 0] == predictions[:, 0]
    print "Accuracy after greedy training :", numpy.mean(correct)

    print 'Running the Fine tune with the cost function stacked_auto_encoder_cost...'
    opt_solution = scipy.optimize.minimize(
        stacked_auto_encoder_cost, all_theta, args=(net_config, lambda_, train_data, train_labels,),
        method='L-BFGS-B', jac=True, options={'maxiter': max_iterations, 'disp': True}
    )
    fine_tuning_theta = opt_solution.x
    joblib.dump(fine_tuning_theta, 'model/FT_theta.pkl')

    print 'Predicting after fine tuning...'
    fine_tuning_theta = joblib.load('model/FT_theta.pkl')
    predictions = stacked_auto_encoder_predict(fine_tuning_theta, net_config, test_data)
    correct = test_labels[:, 0] == predictions[:, 0]
    print "Accuracy after fine tuning :", numpy.mean(correct)

    # --------------------------------------------
    w1 = sae1_theta[0:visible_size * hidden_size1].reshape(hidden_size1, visible_size)
    image_w1 = display_network(w1.transpose())
    fig1 = plt.figure()
    plt.imshow(image_w1, cmap=plt.cm.gray)
    plt.title('First AutoEncoder Weight.')

    w2 = sae2_theta[0:hidden_size2 * hidden_size1].reshape(hidden_size2, hidden_size1)
    image_w2 = display_network(w2.transpose())
    fig2 = plt.figure()
    plt.imshow(image_w2, cmap=plt.cm.gray)
    plt.title('Second AutoEncoder Weight.')
    plt.show()
    # --------------------------------------------

if __name__ == '__main__':
    execute_stacked_auto_encoder()
