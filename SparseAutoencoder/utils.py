#!/usr/bin/python
# coding: utf-8

"""
    ...
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    Author: YuJun <cuteuy@gmail.com>
"""

import time
import array
import struct
import scipy.io
import numpy as np
import scipy.optimize
import matplotlib.pyplot


def normalize_dataset(dataset):
    # Remove mean of dataset
    dataset -= np.mean(dataset)

    # Truncate to +/-3 standard deviations and scale to -1 to 1
    std_dev = 3 * np.std(dataset, ddof=1)
    dataset = np.maximum(np.minimum(dataset, std_dev), -std_dev) / std_dev

    # Rescale from [-1, 1] to [0.1, 0.9]
    dataset = (dataset + 1) * 0.4 + 0.1

    return dataset


def load_dataset(num_patches, patch_side):
    """ 将num_patches块大小为patch_side*patch_side的图片加载至array并返回 """

    images = scipy.io.loadmat('data/IMAGES.mat')
    images = images['IMAGES']

    from scipy.misc import toimage
    toimage(images[:, :, 1]).show()

    # 初始化容器
    dataset = np.zeros((patch_side * patch_side, num_patches))

    # 初始化块选取未知和截取块的图片选取位置
    rand = np.random.RandomState(int(time.time()))
    image_indices = rand.randint(512 - patch_side, size=(num_patches, 2))
    image_number = rand.randint(10, size=num_patches)

    # 抽样、存储、归一化并返回
    for i in xrange(num_patches):
        index1 = image_indices[i, 0]
        index2 = image_indices[i, 1]
        index3 = image_number[i]

        patch = images[index1:(index1+patch_side), index2:(index2+patch_side), index3]
        patch = patch.flatten()
        dataset[:, i] = patch

    return normalize_dataset(dataset)


def load_mnist_images(file_name):
    """ Loads the images from the provided file name """

    image_file = open(file_name, 'rb')  # Open the file

    # Read header information from the file
    head1 = image_file.read(4)
    head2 = image_file.read(4)
    head3 = image_file.read(4)
    head4 = image_file.read(4)

    # Format the header information for useful data
    num_examples = struct.unpack('>I', head2)[0]
    num_rows = struct.unpack('>I', head3)[0]
    num_cols = struct.unpack('>I', head4)[0]

    # Initialize dataset as array of zeros
    dataset = np.zeros((num_rows*num_cols, num_examples))

    # Read the actual image data
    images_raw = array.array('B', image_file.read())
    image_file.close()

    # Arrange the data in columns
    for i in range(num_examples):

        limit1 = num_rows * num_cols * i
        limit2 = num_rows * num_cols * (i + 1)

        dataset[:, i] = images_raw[limit1: limit2]

    # Normalize and return the dataset
    return dataset / 255


# 将转化为稀疏节点的各权重在图片的相应位置展示出来
def visualize_w1(opt_w1, vis_side, hid_side):
    """ Add the weights as a matrix of images """

    figure, axes = matplotlib.pyplot.subplots(nrows=hid_side,
                                              ncols=hid_side)
    index = 0

    for axis in axes.flat:
        """ Add row of weights as an image to the plot """

        image = axis.imshow(opt_w1[index, :].reshape(vis_side, vis_side),
                            cmap=matplotlib.pyplot.cm.gray, interpolation='nearest')
        axis.set_frame_on(False)
        axis.set_axis_off()
        index += 1

    """ Show the obtained plot """

    matplotlib.pyplot.show()
