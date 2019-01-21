"""
This script trains large-scale logistic regression
on serverless systems (AWS Lambda) using the serverless
computing framework developed in pywren and numpywren.
The training data can be generated synthetically or can be 
real-world downloaded from the internet.
The algorithm used is called OverSketched Newton.
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

## Define the block-size and sketching dimension
b = n_features
N = int(15)
d = int(N*b)


def calculate_sketched_sqrt_hessian(id, X_s3_unconv, gamma, hashes, flips, b):
    """
    Calculates sqrt of Hesssian and its OverSketch HS for fat H 
    """
    x = id[0]
    y = id[1]
    X = X_s3_unconv.get_block(0,x)
    a = gamma.get_block(0,0)
    sq_H = np.zeros(X.shape)
    for i2 in range(X.shape[1]):
        sq_H[:,i2] = np.multiply(X[:,i2], np.squeeze(a))

    sq_H = sq_H.T
    
    m,n = sq_H.shape
    hash_local = hashes[y,:]
    flip_local = flips[y,:]
    sketch_block = np.zeros((m, b))
    for i in range(n):
        sketch_block[:, hash_local[i]] += flip_local[i]*sq_H[:,i]
    return sketch_block/np.sqrt(N), id

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
iterable = [(x,y) for x in X_s3_unconv._block_idxs(1) for y in range(N)]
alpha = np.zeros((n_samples,1))
w.put_block(w_loc,0,0)
alpha_test = np.zeros((n_samples_test,1))
training_loss2 = [np.sum(np.log(1+np.exp(-np.multiply(y2, alpha))))/n_samples]
testing_loss2  = [np.sum(np.log(1+np.exp(-np.multiply(y_test2, alpha_test))))/n_samples_test]
t_grad = 0
t_hessian = 0

"""
Iterations for OverSketched Newton start
"""
for ii in range(0,8):
    print("ITERATION {}: ITERATION TIME {:.4f} TRAINING LOSS {:.4f} TEST LOSS {:.4f}".
        format(ii+1, iter_times2[-1], training_loss2[-1], testing_loss2[-1]))
    
    t_start = time.time()    
    beta_loc = np.divide(y2, np.exp(np.multiply(alpha, y2)) + 1)
    beta.put_block(beta_loc,0,0)     
    t_grad_start = time.time() 
    g = StragglerProofLA.coded_mat_vec_mul(X_unconv_coded, beta, num_parity_blocks, coding_length)
    t_grad = t_grad + time.time() - t_grad_start
    # g = X.T.dot(beta_loc)
    # g = StragglerProofLA.recompute_mat_vec_mul(X_s3_unconv.T, beta, thres = 0.95)
    g = (-1/n_samples)*g + 2*cons*w_loc
    
    ## For SECOND-ORDER update, uncomment
    a = np.divide(np.exp(np.multiply(alpha, y2)), np.square(np.exp(np.multiply(alpha, y2)) + 1))
    a = np.sqrt(a)
    gamma.put_block(a,0,0)  
    
    hashes = np.random.randint(0, b, size=(N, n_samples))
    flips = np.random.choice([-1,1], size=(N, n_samples))
    t_hessian_start = time.time()

    futures = pwex.map(lambda x: calculate_sketched_sqrt_hessian(x, X_s3_unconv, gamma, hashes, flips, b), iterable)
    sketch_sqH = np.zeros((n_features, b*N))
    x_shard_size = X_s3_unconv.shard_sizes[1]
    y_shard_size = b
    not_dones = list(range(len(futures)))
    iterable_not_done = deepcopy(iterable)
    while len(not_dones)>=0.05*len(futures):
        fs_dones, fs_not_dones = pywren.wait(futures,2)
        # print ("Number of workers done", len(fs_dones))
        for i,f in enumerate(futures):
            if f in fs_dones and i in not_dones:
                try:
                    x_ord, y_ord = f.result()[1]
                    sketch_sqH[x_ord*x_shard_size:(x_ord+1)*x_shard_size, y_ord*y_shard_size:(y_ord+1)*y_shard_size] = f.result()[0] 
                    not_dones.remove(i)
                    iterable_not_done.remove((x_ord,y_ord))
                except Exception as e:
                    print(e)
                    pass
        time.sleep(2)
        
    t_hessian = t_hessian + time.time() - t_hessian_start
    
    H = sketch_sqH.dot(sketch_sqH.T)
    H = (1/n_samples)*H + 2*cons*np.identity(n_features)
    dd = np.linalg.solve(H, g)
    w_loc = np.subtract(w_loc , dd)
    
    ## For gradient descent
    # w_loc = w_loc - ss*g
    
    w.put_block(w_loc,0,0)
    t_grad_start = time.time() 
    alpha = StragglerProofLA.coded_mat_vec_mul(X_conv_coded, w, num_parity_blocks, coding_length)
    t_grad = t_grad + time.time() - t_grad_start
    # alpha = StragglerProofLA.recompute_mat_vec_mul(X_s3_conv, w, thres = 0.95)
    # alpha = X.dot(w_loc)
    
    t_total = time.time() - t_start
    iter_times2.append(t_total)
    training_loss2.append(np.sum(np.log(1+np.exp(-np.multiply(y2,alpha))))/n_samples)
    alpha_test = X_test.dot(w_loc)
    testing_loss2.append(np.sum(np.log(1+np.exp(-np.multiply(y_test2, alpha_test))))/n_samples_test)
    

m = np.column_stack((iter_times2, training_loss2, testing_loss2))

print("Iteration times, training_loss, testing_loss")
print(m)

