from __future__ import print_function

import os
import time
import json
import argparse
import densenet
import numpy as np
import keras.backend as K

from keras.datasets import cifar10
from keras.optimizers import Adam, SGD
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

def sample_latency_ANN(ann, batch_shape, repeat):
    samples = []

    # drop first run
    ann.predict(np.random.random(batch_shape), batch_size=batch_shape[0])

    for i in range(repeat):
        data_in = np.random.random(batch_shape)
        start_time = time.time()
        ann.predict(data_in, batch_size=batch_shape[0])
        samples.append(time.time() - start_time)
    per_frame_latency = np.array(samples) / batch_shape[0]
    avg_latency_per_frame = np.average(per_frame_latency)
    std_dev_per_frame = np.std(per_frame_latency)
    return(avg_latency_per_frame, std_dev_per_frame)

def profile_densenet(nb_classes,
                img_dim,
                batch_size,
                nb_epoch,
                depth,
                nb_dense_block,
                nb_filter,
                growth_rate,
                dropout_rate,
                learning_rate,
                weight_decay,
                logfile,
                plot_architecture):
    """ Run CIFAR10 experiments

    :param batch_size: int -- batch size
    :param nb_epoch: int -- number of training epochs
    :param depth: int -- network depth
    :param nb_dense_block: int -- number of dense blocks
    :param nb_filter: int -- initial number of conv filter
    :param growth_rate: int -- number of new filters added by conv layers
    :param dropout_rate: float -- dropout rate
    :param learning_rate: float -- learning rate
    :param weight_decay: float -- weight decay
    :param plot_architecture: bool -- whether to plot network architecture

    """

    ###################
    # Construct model #
    ###################

    model = densenet.DenseNet(nb_classes,
                              img_dim,
                              depth,
                              nb_dense_block,
                              growth_rate,
                              nb_filter,
                              dropout_rate=dropout_rate,
                              weight_decay=weight_decay)
    # Model output
    model.summary()

    # Build optimizer
    # opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    opt = SGD(lr=learning_rate, momentum=0.9, nesterov=True)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=["accuracy"])

    if plot_architecture:
        from keras.utils.visualize_util import plot
        plot(model, to_file='./figures/densenet_archi.png', show_shapes=True)

    ####################
    # Network profiling#
    ####################
    batch_shape = (batch_size, 32, 32, 3)
    repeat = 25

    model_latency, model_CI = sample_latency_ANN(model, batch_shape, repeat)
    print(model_latency)

    d_log = {}
    d_log["batch_size"] = batch_size
    d_log["latency"] = model_latency
    d_log["CI"] = model_CI
    d_log["depth"] = depth
    d_log["nb_dense_block"] = nb_dense_block
    d_log["growth_rate"] = growth_rate
    d_log["nb_filter"] = nb_filter

    logfile = 'densenet_B{}_L{}_k{}.json'.format(nb_dense_block, depth, growth_rate)
    json_file = os.path.join('./log', logfile)
    with open(json_file, 'w') as fp:
        json.dump(d_log, fp, indent=4, sort_keys=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Profiling DenseNet')
    parser.add_argument('--nb_classes', default=10, type=int,
                        help='number of classes')
    parser.add_argument('--image_size', default=32, type=int,
                        help='image size')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size')
    parser.add_argument('--nb_epoch', default=30, type=int,
                       help='Number of epochs')
    parser.add_argument('--depth', type=int, default=7,
                        help='Network depth')
    parser.add_argument('--nb_dense_block', type=int, default=1,
                        help='Number of dense blocks')
    parser.add_argument('--nb_filter', type=int, default=16,
                        help='Initial number of conv filters')
    parser.add_argument('--growth_rate', type=int, default=12,
                        help='Number of new filters added by conv layers')
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                        help='Dropout rate')
    parser.add_argument('--learning_rate', type=float, default=1E-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1E-4,
                        help='L2 regularization on weights')
    parser.add_argument('--logfile', type=str, default='latency.json',
                        help='logfile name')
    parser.add_argument('--plot_architecture', type=bool, default=False,
                        help='Save a plot of the network architecture')

    args = parser.parse_args()

    print("Network configuration:")
    for name, value in parser.parse_args()._get_kwargs():
        print(name, value)

    list_dir = ["./log", "./figures"]
    for d in list_dir:
        if not os.path.exists(d):
            os.makedirs(d)

    img_dim = (args.image_size, args.image_size, 3)

    profile_densenet(args.nb_classes,
                img_dim,
                args.batch_size,
                args.nb_epoch,
                args.depth,
                args.nb_dense_block,
                args.nb_filter,
                args.growth_rate,
                args.dropout_rate,
                args.learning_rate,
                args.weight_decay,
                args.logfile,
                args.plot_architecture)
