"""
This script trains large-scale logistic regression
on serverless systems (AWS Lambda) using the serverless
computing framework developed in pywren and numpywren.
The training data can be generated synthetically or can be 
real-world downloaded from the internet.
The algorithm used is Newton's method.
"""

from __future__ import print_function
import sys

## Import the functions required for coded computing of gradient 
sys.path.insert(0, './FunctionsCodeDecode')

import os
import numpy as np
import random
import pandas as pd
from sklearn import  preprocessing
from sklearn.model_selection import train_test_split
import itertools
import math
from scipy.sparse import csr_matrix
import make_coding_function
import decode2D
from copy import deepcopy
import StragglerProofLA


import pywren
import time
import numpywren
from numpywren import matrix, matrix_utils 
from numpywren import binops
from numpywren.matrix_init import shard_matrix, local_numpy_init, reshard_down
from numpywren.matrix_utils import chunk
from generate_logistic_data import toy_logistic_data



"""
Define the dimensions of data in S3
"""
n_features = 3000
n_samples = 300000
n_samples_test = int(0.25*n_samples)


## Define number of processors to use while calculating the gradient

n_procs=60  

## Define number of parity blocks to use for coded computation
num_parity_blocks=6 #Make num_parity blocks close to sqrt(n_procs) for efficiency


"""
Define numpywren BigMatrix and make sure the data
has been uploaded to S3 cloud storage
(done using the script upload_data_to_s3)
"""

X_s3_conv = matrix.BigMatrix("logistic_synthetic_data_{0}_{1}_{2}".format(n_samples, n_features, n_procs), shape=(n_samples, n_features), 
                     shard_sizes=(n_samples//n_procs, n_features), write_header=True)


X_s3_test = matrix.BigMatrix("logistic_epsilon_test_data_{0}_{1}".format(n_samples_test, n_features), shape=(n_samples_test, n_features), 
                     shard_sizes=(n_samples_test, n_features), write_header=True)

y_s3_conv = matrix.BigMatrix("logistic_synthetic_data_y_{0}_{1}".format(n_samples, n_procs), shape=(n_samples,), 
                     shard_sizes=(n_samples//n_procs,), write_header=True)

y_s3_test = matrix.BigMatrix("logistic_epsilon_labels_test_y_{0}".format(n_samples_test), shape=(n_samples_test,), 
                     shard_sizes=(n_samples_test,), write_header=True)

X_s3_unconv = matrix.BigMatrix("logistic_synthetic_data_{0}_{1}_{2}".format(n_samples, n_features, n_procs), shape=(n_samples, n_features), 
                     shard_sizes=(n_samples, int(np.ceil(n_features/n_procs))), write_header=True)

assert(X_s3_conv.block_idxs_not_exist == [])
assert(y_s3_conv.block_idxs_not_exist == [])
assert(X_s3_unconv.block_idxs_not_exist == [])

print("Data exists in cloud, Encoding of matrices starts")



pwex = pywren.lambda_executor()

cons = 1.0/n_samples

"""
Obtaining training and test data to
calculate the training and testing error locally 
"""

X = X_s3_conv.numpy()
X_test = X_s3_test.numpy()
y2 = y_s3_conv.numpy()
y_test = y_s3_test.numpy()

## Encode the training matrix
X_conv_coded = make_coding_function.code_2D(X_s3_conv, num_parity_blocks, thres=0.9)

X_unconv_coded = make_coding_function.code_2D(X_s3_unconv.T, num_parity_blocks, thres=0.9)


coding_length = int(np.ceil(len(X_s3_conv._block_idxs(0))/num_parity_blocks))

print ("Encoding of matrices done, Optimization starts")

Y = X_s3_unconv.T

coding_length = int(np.ceil(len(X_s3_conv._block_idxs(0))/num_parity_blocks))


"""
This function is mapped over serverless
workers for Hessian calculation.
"""
def calculate_hessian(id):
    i = id[0]
    j = id[1]
    a = gamma.get_block(0,0)
    X = X_s3_unconv.get_block(0,j)
    sq_H = np.zeros(X.shape)
    for i2 in range(X.shape[1]):
        sq_H[:,i2] = np.multiply(X[:,i2], np.squeeze(a))
    X1 = X_s3_unconv.get_block(0,i)
    return np.dot(X1.T,sq_H), id

y2 = y2.reshape((n_samples,1))
y_test2 = y_test.reshape((n_samples_test,1))
iter_times2 = [0]
w_loc = np.zeros((n_features,1))
w = matrix.BigMatrix("w_t_{0}".format(n_features), shape=w_loc.shape, 
                     shard_sizes=w_loc.shape, write_header=True)
beta = matrix.BigMatrix("beta_t_{0}".format(n_samples), shape=(n_samples,1), 
                     shard_sizes=(n_samples,1), autosqueeze=False, write_header=True)
gamma = matrix.BigMatrix("gamma_t_{0}".format(n_samples), shape=(n_samples,1), 
                     shard_sizes=(n_samples,1), autosqueeze=False, write_header=True)
iterable = [(x,y) for x in X_s3_unconv._block_idxs(1) for y in X_s3_unconv._block_idxs(1)]
alpha = np.zeros((n_samples,1))
w.put_block(w_loc,0,0)
alpha_test = np.zeros((n_samples_test,1))
training_loss2 = [np.sum(np.log(1+np.exp(-np.multiply(y2, alpha))))/n_samples]
testing_loss2  = [np.sum(np.log(1+np.exp(-np.multiply(y_test2, alpha_test))))/n_samples_test]

"""
Iterations for Newton's method start
"""
for ii in range(8):
    print("ITERATION {}: ITERATION TIME {:.4f} TRAINING LOSS {:.4f} TEST LOSS {:.4f}".
        format(ii+1, iter_times2[-1], training_loss2[-1], testing_loss2[-1]))
    t_start = time.time()
    
    beta_loc = np.divide(y2, np.exp(np.multiply(alpha, y2)) + 1)
    beta.put_block(beta_loc,0,0)     
    g = StragglerProofLA.coded_mat_vec_mul(X_unconv_coded, beta, num_parity_blocks, coding_length)
    # g = X.T.dot(beta_loc)
    g = (-1/n_samples)*g + 2*cons*w_loc
    
    ## For SECOND-ORDER update, uncomment
    a = np.divide(np.exp(np.multiply(alpha, y2)), np.square(np.exp(np.multiply(alpha, y2)) + 1))
    gamma.put_block(a,0,0)
    futures = pwex.map(calculate_hessian, iterable)
    
    """
    Uncomment following to calculate the 
    Hessian locally for verification purposes.
    """
    # sq_H = np.zeros(X.shape)
    # for i in range(n_samples):
    #     sq_H[i,:] = np.sqrt(a[i])*X[i,:]
    # H = np.dot(sq_H.T, sq_H)    

    H = np.zeros((n_features, n_features))
    shard_size = X_s3_unconv.shard_sizes[1]
    not_dones = list(range(len(futures)))
    iterable_not_done = deepcopy(iterable)
    while len(not_dones)>=0.02*len(futures):
        fs_dones, fs_not_dones = pywren.wait(futures,2)
        for i,f in enumerate(futures):
            if f in fs_dones and i in not_dones:
                try:
                    x_ord, y_ord = f.result()[1]
                    H[x_ord*shard_size:(x_ord+1)*shard_size, y_ord*shard_size:(y_ord+1)*shard_size] = f.result()[0] 
                    not_dones.remove(i)
                    iterable_not_done.remove((x_ord,y_ord))
                except Exception as e:
                    print(e)
                    pass
        time.sleep(2)
    
    futures2 = pwex.map(calculate_hessian, iterable_not_done)
    pywren.wait(futures2)
    for f in futures2:
        try:
            x_ord, y_ord = f.result()[1]
            H[x_ord*shard_size:(x_ord+1)*shard_size, y_ord*shard_size:(y_ord+1)*shard_size] = f.result()[0] 
        except Exception as e:
            print(e)
            pass
    
    H = (1/n_samples)*H + 2*cons*np.identity(n_features)
    dd = np.linalg.solve(H, g)
    w_loc = np.subtract(w_loc , dd)
    
    ## For gradient descent
    # w_loc = w_loc - ss*g

    w.put_block(w_loc,0,0)
    
    alpha = StragglerProofLA.coded_mat_vec_mul(X_conv_coded, w, num_parity_blocks, coding_length)
    # alpha = X.dot(w_loc)
    
    t_total = time.time() - t_start
    iter_times2.append(t_total)
    training_loss2.append(np.sum(np.log(1+np.exp(-np.multiply(y2,alpha))))/n_samples)
    alpha_test = X_test.dot(w_loc)
    testing_loss2.append(np.sum(np.log(1+np.exp(-np.multiply(y_test2, alpha_test))))/n_samples_test)

m = np.column_stack((iter_times2, training_loss2, testing_loss2))

print("Iteration times, training_loss, testing_loss")
print(m)

