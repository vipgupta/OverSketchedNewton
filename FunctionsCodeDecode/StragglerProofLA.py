import pywren
import time
import numpywren
from numpywren import matrix, matrix_utils
from numpywren import binops
from numpywren.matrix_init import shard_matrix, local_numpy_init, reshard_down
import numpy as np
import decode2D
from copy import deepcopy

def coded_mat_vec_mul(A_coded_2D, x, num_parity_blocks, coding_length):
    
    def coded_mat_vec_mul(id):
        shard_size = A_coded_2D.shard_sizes[1]
        reduce_idxs = A_coded_2D._block_idxs(axis=1)
        x_loc = x.get_block(0,0)
        Ax_block = None
        for r in reduce_idxs:
            block1 = A_coded_2D.get_block(id, r)
            sidx = r*shard_size
            eidx = (r+1)*shard_size
            x_block = x_loc[sidx:eidx]
            if (Ax_block is None):
                Ax_block = block1.dot(x_block)
            else:
                Ax_block = Ax_block + block1.dot(x_block)       
        return Ax_block

    shard_size = A_coded_2D.shard_sizes[0]
    n_coded_procs = len(A_coded_2D._block_idxs(0))
    len_A_coded = n_coded_procs - coding_length - 1
    
    pwex = pywren.lambda_executor()
    futures = pwex.map(coded_mat_vec_mul, range(n_coded_procs))   
    Ax = np.zeros((A_coded_2D.shape[0],1))
    bitmask = np.ones((coding_length + 1, num_parity_blocks + 1))
    not_done = list(range(n_coded_procs))
    while decode2D.cant_be_decoded_with_bitmask(deepcopy(bitmask)):
        fs_dones, fs_not_dones = pywren.wait(futures,2)
        for (id,f) in enumerate(futures):
            if f in fs_dones and id in not_done:
                # print("Worker done", id)
                try:
                    Ax[id*shard_size:(id+1)*shard_size] = f.result()
                    i, j = decode2D.ind1Dto2D(id, len_A_coded, num_parity_blocks)
                    bitmask[i, j] = 0
                    not_done.remove(id)
                except Exception as e:
                    print(e)
                    pass
    # print("1: Decoding not dones", not_done)
    Ax = decode2D.decode_vector_with_bitmask(Ax, bitmask, num_parity_blocks, 
                                                    len_A_coded, shard_size, coding_length)
    return Ax



def recompute_mat_vec_mul(A_coded_2D, x, thres=0.95):
    
    def shard_mat_vec_mul(id):
        shard_size = A_coded_2D.shard_sizes[1]
        reduce_idxs = A_coded_2D._block_idxs(axis=1)
        x_loc = x.get_block(0,0)
        Ax_block = None
        for r in reduce_idxs:
            block1 = A_coded_2D.get_block(id, r)
            sidx = r*shard_size
            eidx = (r+1)*shard_size
            x_block = x_loc[sidx:eidx]
            if (Ax_block is None):
                Ax_block = block1.dot(x_block)
            else:
                Ax_block = Ax_block + block1.dot(x_block)       
        return Ax_block, id

    shard_size = A_coded_2D.shard_sizes[0]
    n_coded_procs = len(A_coded_2D._block_idxs(0))
    
    pwex = pywren.lambda_executor()
    futures = pwex.map(shard_mat_vec_mul, range(n_coded_procs))   
    Ax = np.zeros((A_coded_2D.shape[0],1))
    not_done = list(range(n_coded_procs))
    fs_dones = []
    f_result_dones = []
    while len(fs_dones) < thres*n_coded_procs:
        fs_dones, fs_not_dones = pywren.wait(futures,2)
        for f in list(set(fs_dones) - set(f_result_dones)):
            # print("Worker done", id)
            f_result_dones.append(f)
            try: 
                result = f.result()
                id = result[1]
                Ax[id*shard_size:(id+1)*shard_size] = result[0]
                not_done.remove(id)
            except Exception as e:
                #print(e)
                pass
        time.sleep(2)
    print("Recomputing not dones", not_done)
    futures2 = pwex.map(shard_mat_vec_mul, not_done) 
    f_result_dones2 = []
    while not_done!=[]:
        fs_dones2, fs_not_dones2 = pywren.wait(futures2,3)
        for f in list(set(fs_dones2) - set(f_result_dones2)):
            f_result_dones2.append(f)
            try:
                result = f.result()
                id = result[1]
                if id in not_done:
                    print("Recomputed", id)
                    Ax[id*shard_size:(id+1)*shard_size] = result[0]
                    not_done.remove(id)
            except Exception as e:
                #print(e)
                pass
        time.sleep(2)
        fs_dones, fs_not_dones = pywren.wait(futures,3)
        for f in list(set(fs_dones) - set(f_result_dones)):
            f_result_dones.append(f)
            try:
                result = f.result()
                id = result[1]
                if id in not_done:
                    print("Straggler computed", id)
                    Ax[id*shard_size:(id+1)*shard_size] = result[0]
                    not_done.remove(id)
            except Exception as e:
                #print(e)
                pass
        time.sleep(2)
        if fs_not_dones2==[] and fs_not_dones==[]:
            print ("NOT DONE", not_done)
            break
    print("Recomputing done")
    return Ax