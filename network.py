# -*- coding: utf-8 -*-
# @File    : network.py
# @Author  : zhengdongjian
# @Time    : 2017/3/22 下午11:14
# @Desp    :
import random
import numpy as np


def sigmoid(z):
    """
    sigmoid函数
    :param z:
    :return:
    """
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """
    sigmoid(z) * (1 - sigmoid(z))
    :param z:
    :return:
    """
    tmp = sigmoid(z)
    return tmp * (1 - tmp)


class Network(object):
    """
    神经网络对象
    """
    def __init__(self, sizes):
        """
        初始化
        :param sizes: 网络各层神经元数量list
        """
        self.num_layers = len(sizes)  # 层数
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def __repr__(self):
        """
        repr
        :return:
        """
        return ('Network{layers=' + repr(self.num_layers) + ';\nbiases=' + repr(self.biases) + ';\nweights=' + repr(self.weights)
                + '}')

    # @staticmethod
    # def sigmoid(z):
    #     return sigmoid(z)

    def feed_forward(self, a):
        """
        前馈方法
        :param a: 输入
        :return: 输出
        """
        for bias, weight in zip(self.biases, self.weights):
            a = sigmoid(np.dot(weight, a) + bias)  # dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])
        return a

    def back_prop(self, x, y):
        """
        反向传播
        :param x:
        :param y:
        :return:
        """
        nabla_bias = [np.zeros(bias.shape) for bias in self.biases]
        nabla_weight = [np.zeros(weight.shape) for weight in self.weights]
        # 前向传播，得到计算各层的输入及激活值
        activation = x
        activations = [activation]
        zs = []  # z向量列表
        for bias, weight in zip(self.biases, self.weights):
            z = np.dot(weight, activation) + bias
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # 反向传递，计算最后一层的输出误差，然后将该误差反向传播
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_bias[-1] = delta
        # noinspection PyTypeChecker
        nabla_weight[-1] = np.dot(delta, activations[-2].transpose())
        # l=1表示最后一层
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-(l - 1)].transpose(), delta) * sp
            nabla_bias[-l] = delta
            nabla_weight[-l] = np.dot(delta, activations[-(l + 1)].transpose())
        return nabla_bias, nabla_weight

    def evaluate(self, test_data):
        """
        计算输入test_data对应的输出
        :param test_data:
        :return:
        """
        test_results = [(np.argmax(self.feed_forward(x)), y) for x, y in test_data]
        # from platform import python_version_tuple
        # if python_version_tuple()[0] == '2':
        #     return sum(1 if x == y else 0 for x, y in test_results)
        return sum(int(x == y) for x, y in test_results)

    def cost_derivative(self, output_activations, y):
        """
        代价函数
        :param output_activations:
        :param y:
        :return:
        """
        return output_activations - y

    def update_mini_batch(self, mini_batch, eta):
        """
        根据mini_batch更新网络权重和偏置
        :param mini_batch: 随机选取的mini_batch
        :param eta: 学习率
        :return:
        """
        nabla_bias = [np.zeros(bias.shape) for bias in self.biases]
        nabla_weight = [np.zeros(weight.shape) for weight in self.weights]
        for x, y in mini_batch:
            delta_nabla_bias, delta_nabla_weight = self.back_prop(x, y)
            nabla_bias = [nb + dnb for nb, dnb in zip(nabla_bias, delta_nabla_bias)]
            nabla_weight = [nw + dnw for nw, dnw in zip(nabla_weight, delta_nabla_weight)]
        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_weight)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_bias)]

    def sgd(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """
        梯度下降方法
        :type training_data: tuples(x, y)
        :param training_data: 输入，训练数据
        :param epochs: 轮数，one epoch = numbers of iterations = N = 训练样本的数量/batch size
        :param mini_batch_size: batch大小
        :param eta: 学习率
        :param test_data: 测试数据，若传入，则在每一轮(epoch)结束后用该测试数据评价一次网络
        :return: 输出tuples(x, y)
        """
        n_test = None
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(1, epochs + 1):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print('Epoch {}: {} / {}'.format(j, self.evaluate(test_data), n_test))
            else:
                print('Epoch {} complete'.format(j))


if __name__ == '__main__':
    from mnist_loader import load_data_wrapper
    training_dat, validation_dat, test_dat = load_data_wrapper()
    net = Network([784, 30, 10])
    net.sgd(training_dat, 30, 10, 1.0, test_dat)
