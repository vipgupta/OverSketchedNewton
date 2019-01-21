"""
This script generates synthetic (or real-world)
dataset and uploads it to S3
"""

import os
import numpy as np
import random
import pandas as pd
from sklearn import  preprocessing
from sklearn.model_selection import train_test_split
import itertools
import math
from scipy.sparse import csr_matrix


import pywren
import time
import numpywren
from numpywren import matrix, matrix_utils 
from numpywren import binops
from numpywren.matrix_init import shard_matrix, local_numpy_init, reshard_down
from numpywren.matrix_utils import chunk
from generate_logistic_data import toy_logistic_data

"""
To import training and testing real dataset EPSILON, uncomment
the following. You need to download the dataset from 
https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/
in the input_dir
"""

# from tick.dataset import fetch_tick_dataset
# sys.path.insert(0, '/home/ec2-user/FunctionsCodeDecode/real_datasets')
# input_dir = "/home/ec2-user/FunctionsCodeDecode/real_datasets/"

# train_set = fetch_tick_dataset(input_dir + 'epsilon_normalized.bz2')
# test_set = fetch_tick_dataset(input_dir + 'epsilon_normalized.t.bz2')


# X = train_set[0]
# X = X.toarray()
# y = train_set[1]
# y2 = y

# X_test = test_set[0]
# y_test = test_set[1]
# X_test = X_test.toarray()
# n_features = X.shape[1]
# n_samples = X.shape[0]
# n_samples_test = X_test.shape[0]


"""
To generate synthetic dataset, uncomment
"""
n_features = 3000
n_samples = 300000
n_samples_test = int(0.25*n_samples)
X, X_test, y2, y_test = toy_logistic_data(n_samples, n_samples_test, n_features)
y = y2


## Define number of processors to use while calculating the gradient

n_procs=60  

## Define numpywren BigMatrix and upload data to S3 cloud storage

X_s3_conv = matrix.BigMatrix("logistic_synthetic_data_{0}_{1}_{2}".format(n_samples, n_features, n_procs), shape=(n_samples, n_features), 
                     shard_sizes=(n_samples//n_procs, n_features), write_header=True)
shard_matrix(X_s3_conv, X, overwrite=True)

X_s3_unconv = matrix.BigMatrix("logistic_synthetic_data_{0}_{1}_{2}".format(n_samples, n_features, n_procs), shape=(n_samples, n_features), 
                     shard_sizes=(n_samples, int(np.ceil(n_features/n_procs))), write_header=True)
shard_matrix(X_s3_unconv, X, overwrite=True)

X_s3_test = matrix.BigMatrix("logistic_epsilon_test_data_{0}_{1}".format(n_samples_test, n_features), shape=(n_samples_test, n_features), 
                     shard_sizes=(n_samples_test, n_features), write_header=True)
shard_matrix(X_s3_test, X_test, overwrite=True)

y_s3_conv = matrix.BigMatrix("logistic_synthetic_data_y_{0}_{1}".format(n_samples, n_procs), shape=(n_samples,), 
                     shard_sizes=(n_samples//n_procs,), write_header=True)
shard_matrix(y_s3_conv, y2, overwrite=True)

y_s3_test = matrix.BigMatrix("logistic_epsilon_labels_test_y_{0}".format(n_samples_test), shape=(n_samples_test,), 
                     shard_sizes=(n_samples_test,), write_header=True)
shard_matrix(y_s3_test, y_test, overwrite=True)