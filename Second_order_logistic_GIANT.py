"""
This script trains large-scale logistic regression
on serverless systems (AWS Lambda) using the serverless
computing framework developed in pywren and numpywren.
The training data can be generated synthetically or can be 
real-world downloaded from the internet.
The algorithm used is called GIANT developed in the 
following paper
https://arxiv.org/abs/1709.03528 
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

"""
Define number of parity blocks to use for coded computation
"""
num_parity_blocks = 6 #Make num_parity blocks close to sqrt(n_procs) for efficiency


"""
Define numpywren BigMatrix and make sure the data
has been uploaded to S3 cloud storage
(done using the script upload_data_to_s3).
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

print("Data exists in cloud, Optimization starts")



pwex = pywren.lambda_executor()

cons = 1.0/n_samples

"""
Obtaining training and test data to
calculate the training and testing error locally. 
"""

X = X_s3_conv.numpy()
X_test = X_s3_test.numpy()
y2 = y_s3_conv.numpy()
y_test = y_s3_test.numpy()


"""
This function is mapped over serverless
workers for gradient calculation.
"""
def partial_grad(id, X_s3, y_s3, w_s3, n_procs, s):
    i = ((s+1)*id)%n_procs
    g = None
    w = w_s3.get_block(0)
    for t in range(s+1):
        X = X_s3.get_block(i+t,0)
        y = y_s3.get_block(i+t)
        predy = X.dot(w)
        if g is None:
            g = X.T.dot(np.divide(y, np.exp(np.multiply(predy, y)) + 1))
        else:
            g = g + X.T.dot(np.divide(y, np.exp(np.multiply(predy, y)) + 1))
    return g

"""
This function is mapped over serverless
workers for second-order descent direction
calculation.
"""
def newton_dir(id, X_s3, y_s3, w_s3, g_s3, n_procs, s):
    i = ((s+1)*id)%n_procs
    dd = None
    g = g_s3.get_block(0)
    w = w_s3.get_block(0)
    for t in range(s+1):
        print("i", i, "t", t)
        X = X_s3.get_block(i+t,0)  
        y = y_s3.get_block(i+t)
        alpha = X.dot(w)
        a = np.divide(np.exp(np.multiply(alpha, y)), np.square(np.exp(np.multiply(alpha, y)) + 1))
        
        sq_H = np.zeros(X.shape)
        for iii in range(X.shape[0]):
            sq_H[iii,:] = np.sqrt(a[iii])*X[iii,:]
        H = X.T.dot(X)
        H = H*n_procs/(n_samples) + 2*cons*np.identity(n_features)
        if dd is None:
            dd = np.linalg.solve(H,g)
        else:
            dd = dd+np.linalg.solve(H,g)
    return dd


def func(w):
    predy = X.dot(w)
    return np.sum(np.log(1+np.exp(-np.multiply(y2,predy))))/n_samples + cons*(np.linalg.norm(w)**2)


"""
Backtracking line-search can be used
to find a good step-size for gradient
descent
"""
def backtracking(func, gradf, descent_dir, w_prev):
    ss = 100;
    beta = 0.6;
    alpha = 0.3;
    count = 0;
    while func(w_prev + ss*descent_dir) > func(w_prev) + alpha*ss*np.dot(gradf.T,descent_dir):
        ss = beta*ss;
        count = count+1;
        if count>200:
            break
    return ss

pwex = pywren.lambda_executor()


ss = 1
cons = 1.0/n_samples


"""
For uncoded do s=0, where s denotes the 
number of expected stragglers, otherwise
use a positive integer value to do gradient
coding.
"""
s=0  


y2 = np.squeeze(y2)
w_loc = np.zeros(n_features)
w = matrix.BigMatrix("w_t_{0}".format(n_features), shape=w_loc.shape, 
                     shard_sizes=w_loc.shape, autosqueeze=False, write_header=True)
g_s3 = matrix.BigMatrix("g_t_{0}".format(n_features), shape=w_loc.shape, 
                     shard_sizes=w_loc.shape, autosqueeze=False, write_header=True)
w.put_block(w_loc,0)
iter_times = [0]
predy = np.zeros(n_samples)
predy_test = np.zeros(n_samples_test)
training_loss = [np.sum(np.log(1+np.exp(-np.multiply(y2, predy))))/n_samples]
testing_loss  = [np.sum(np.log(1+np.exp(-np.multiply(y_test, predy_test))))/n_samples_test]


"""
Iterations for solving the optimization
problem using GIANT start here
"""
for ii in range(39):
    print("ITERATION {}: TRAINING LOSS {:.4f}, TESTING LOSS {:.4f}, ITERATION TIME {:.3f}"
          .format(ii,training_loss[-1], testing_loss[-1], iter_times[-1]))
    t_start = time.time()
    # g = X.T.dot(np.divide(y2, np.exp(np.multiply(predy, y2)) + 1))

    futures = pwex.map(lambda x: partial_grad(x, X_s3_conv, y_s3_conv, w, n_procs, s), range(n_procs))
    g = None        
    ## STRAGGLER mitigation
    num_needed = int(n_procs/(s+1))
    not_done = list(range(num_needed))
#     while not_done!=[]:
    while len(not_done)>=0.1*len(futures):              ##For stochastic updates
        fs_dones, fs_not_dones = pywren.wait(futures,2)
        # print ("Number of workers done", len(fs_dones))
        for i,f in enumerate(futures):
            if (i%num_needed) in not_done and f in fs_dones:
                try:
                    if g is None:
                        g = f.result()
                    else:
                        g = g + f.result() 
                    not_done.remove(i%num_needed)
                except Exception as e:
                    print(e)
                    if s==0:
                        not_done.remove(i%num_needed)
                    pass                
        time.sleep(2)

    g = g*(num_needed/(num_needed-len(not_done)))   
    g = (-1/n_samples)*g + 2*cons*w_loc
    g_s3.put_block(g,0)
    
    ## Approximate Second order GIANT
    futures2 = pwex.map(lambda x: newton_dir(x, X_s3_conv, y_s3_conv, w, g_s3, n_procs, s), range(n_procs))
    dd = None  

    ## STRAGGLER mitigation: Gradient coding based GIANT
    num_needed = int(n_procs/(s+1))
    not_done = list(range(num_needed))
#     while not_done!=[]:
    while len(not_done)>=0.1*len(futures2):              ## For ignoring stragglers scheme
        fs_dones, fs_not_dones = pywren.wait(futures2,2)
        # print ("Number of workers done", len(fs_dones))
        for i,f in enumerate(futures2):
            if (i%num_needed) in not_done and f in fs_dones:
                try:
                    if dd is None:
                        dd = f.result()
                    else:
                        dd = dd + f.result() 
                    not_done.remove(i%num_needed)
                except Exception as e:
                    print(e)
                    if s==0:
                        not_done.remove(i%num_needed)
                    pass   
        time.sleep(2)
        
    dd = dd*(num_needed/(num_needed-len(not_done)))     
    dd = dd/(n_procs)
    
    """
    Use backtracking only for gradient descent
    """
    # ss = backtracking(func, g, -dd, w_loc)
    # print ("step-size", ss)

    ss = 1
    w_loc = w_loc - ss*dd
    w.put_block(w_loc,0)
    iter_times.append(time.time() - t_start)
    predy = X.dot(w_loc)
    predy_test = X_test.dot(w_loc)
    
    training_loss.append(np.sum(np.log(1+np.exp(-np.multiply(y2, predy))))/n_samples)
    testing_loss.append(np.sum(np.log(1+np.exp(-np.multiply(y_test, predy_test))))/n_samples_test)
    

m = np.column_stack((iter_times, training_loss, testing_loss))
print("Iteration times, training_loss, testing_loss")
print(m)

