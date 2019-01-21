import pywren
import time
import numpywren
from numpywren import matrix, matrix_utils
from numpywren import binops
from numpywren.matrix_init import shard_matrix, local_numpy_init, reshard_down
import numpy as np

def code_2D(A, num_parity_blocks, thres=1):
    assert(len(A._block_idxs(0))%num_parity_blocks == 0)
    shard_size = A.shard_sizes[0]
    coded_shape = (A.shape[0]+num_parity_blocks*A.shard_sizes[0], A.shape[1])
    coding_length = int(np.ceil(len(A._block_idxs(0))/num_parity_blocks))
    coding_fn2D = make_coding_function2D(A, coding_length)

    coded_2D_shape = (A.shape[0]+(coding_length + 1 + num_parity_blocks)*A.shard_sizes[0], A.shape[1])
    A_coded_2D = matrix.BigMatrix(A.key + "CODED2D_{0}_{1}_{2}".format(A.shape[0], shard_size, num_parity_blocks), 
        shape=coded_2D_shape, shard_sizes=A.shard_sizes, write_header=True, parent_fn=coding_fn2D)

    # if list(set(A_coded_2D.block_idxs_not_exist) - set(A.block_idxs_exist)) == []:
    #     return A_coded_2D

    last_block = max(A._block_idxs(0))
    columns = A_coded_2D._block_idxs(1)
    rows = A_coded_2D._block_idxs(0)
    to_read = []
    blocks_exist = A_coded_2D.block_idxs_exist
    for row in rows:
        if (row <= last_block): continue
        for column in columns:
            if (row,column) in blocks_exist:
                continue
            else:
                to_read.append((row,column))    

    print("Number of parity blocks", len(to_read))   

    num_parities_1D = coding_length*len(A._block_idxs(1))
    to_read_phase1 = to_read[0:num_parities_1D]
    to_read_phase2 = to_read[num_parities_1D:]

    def get_block_wrapper(x):
        A_coded_2D.get_block(*x)
        return 0
    
    #### For 2D ENCODING of A, uncomment
    pwex = pywren.lambda_executor()
    t_enc1 = time.time()
    futures2 = pwex.map(get_block_wrapper, to_read_phase1)
    result_count=0
    fs_dones = []
    while (result_count<thres*len(to_read_phase1)):
        fs_dones, fs_notdones = pywren.wait(futures2, 2)
        result_count = len(fs_dones)
        print(result_count)
        time.sleep(3)
    for f in fs_dones:
        try:
            f.result()
        except Exception as e:
            print(e)
            pass
    t_enc1 =  time.time() - t_enc1
    print ("Encoding phase 1 time", t_enc1)

    t_enc2 = time.time()
    futures2 = pwex.map(get_block_wrapper, to_read_phase2)
    result_count=0
    while (result_count<thres*len(to_read_phase2)):
        fs_dones, fs_notdones = pywren.wait(futures2, 2)
        result_count = len(fs_dones)
        print(result_count)
        time.sleep(3)
    for f in fs_dones:
        try:
            f.result()
        except Exception as e:
            print(e)
            pass
    t_enc2 =  time.time() - t_enc2
    print ("Encoding phase 2 time", t_enc2)
    print ("Total ENCODING time", t_enc1 + t_enc2)
    
    # a = list(set(A_coded_2D.block_idxs_not_exist) - set(A.block_idxs_exist))
    # print("Still to encode", a)
    return A_coded_2D

def make_coding_function2D(X, coding_length):
    def chunk(l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]
    async def coding_function(self,loop,i,j):
        X_sum = None
        num_parity_blocks = int(len(X._block_idxs(0))/coding_length)
        coding_chunks = list(chunk(sorted(X._block_idxs(0)), num_parity_blocks))
        if (i <= max(X._block_idxs(0))):
            return X.get_block(i,j)
        elif (i < len(X._block_idxs(0)) + num_parity_blocks):
            left = i - len(X._block_idxs(0))
            for c in range(coding_length):
                t = c*num_parity_blocks + left
                if (X_sum is None):
                    X_sum = X.get_block(t,j)
                else:
                    X_sum = X_sum + X.get_block(t,j) 
            self.put_block(X_sum, i, j)   
            return X_sum
        elif (i < len(X._block_idxs(0)) + num_parity_blocks + coding_length):
            left = i - len(X._block_idxs(0)) - num_parity_blocks
            for c in coding_chunks[left]:
                if (X_sum is None):
                    X_sum = X.get_block(c,j)
                else:
                    X_sum = X_sum + X.get_block(c,j) 
            self.put_block(X_sum, i, j)
            return X_sum
        elif (i == len(X._block_idxs(0)) + num_parity_blocks + coding_length):
            for c in range(num_parity_blocks):
                t = c + len(X._block_idxs(0))
                if (X_sum is None):
                    X_sum = self.get_block(t,j)
                else:
                    X_sum = X_sum + self.get_block(t,j) 
            self.put_block(X_sum, i, j)   
            return X_sum
        else:
            print ("ERROR: Encoding something not already signified")
            return None
    return coding_function