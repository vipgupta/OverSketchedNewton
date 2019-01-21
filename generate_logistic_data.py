import numpy as np

def generate_random_matrix_normal(mu1, mu2, n_rows, n_cols):
    ctrmu2 = np.random.binomial(n_rows,0.5)
    ctrmu1 = n_rows - ctrmu2 
    mfac = 10/np.sqrt(n_cols)
    return np.concatenate((np.add(mfac*np.random.standard_normal((ctrmu1, n_cols)), mu1), np.add(mfac*np.random.standard_normal((ctrmu2, n_cols)), mu2)))

def generate_random_binvec(n):
    return np.array([np.random.randint(2)*2-1 for x in range(n)])


def toy_logistic_data_old(n_rows, n_test, n_cols, alpha=0.1):

    random_beta = generate_random_binvec(n_cols)
    mu1 = np.multiply(alpha/n_cols, random_beta)
    mu2 = np.multiply(-alpha/n_cols, random_beta)
    label_vector=np.ndarray(n_rows)

    random_matrix = generate_random_matrix_normal(mu1, mu2, n_rows, n_cols)
        
    
    prob_vals = np.divide(1.,np.add(1,np.exp(-np.dot(random_matrix, random_beta))))
    label_vector= np.add(np.multiply(2,np.random.binomial(1,prob_vals)),-1)

    random_matrix_test = generate_random_matrix_normal(mu1, mu2, n_test, n_cols)
    prob_vals = np.divide(1.,np.add(1,np.exp(-np.dot(random_matrix_test, random_beta))))
    label_vector_test = np.add(np.multiply(2,np.random.binomial(1,prob_vals)),-1)

    print("Data Generation Finished.")

    return random_matrix, random_matrix_test, label_vector, label_vector_test

def toy_logistic_data(num_examples, num_examples_test, input_size, weights_prior_stddev=10.0):
    """
    Generates synthetic data for binary classification.
    Args:
    num_examples: The number of samples to generate (scalar Python `int`).
    input_size: The input space dimension (scalar Python `int`).
    weights_prior_stddev: The prior standard deviation of the weight
      vector. (scalar Python `float`).
    Returns:
    random_weights: Sampled weights as a Numpy `array` of shape
      `[input_size]`.
    random_bias: Sampled bias as a scalar Python `float`.
    design_matrix: Points sampled uniformly from the cube `[-1,
       1]^{input_size}`, as a Numpy `array` of shape `(num_examples,
       input_size)`.
    labels: Labels sampled from the logistic model `p(label=1) =
      logistic(dot(inputs, random_weights) + random_bias)`, as a Numpy
      `int32` `array` of shape `(num_examples, 1)`.
    """
    random_weights = weights_prior_stddev * np.random.randn(input_size)
    random_bias = np.random.randn()
    design_matrix = np.random.rand(num_examples, input_size) * 2 - 1
    design_matrix_test = np.random.rand(num_examples_test, input_size) * 2 - 1
    logits = np.reshape(np.dot(design_matrix, random_weights) + random_bias, (-1, 1))
    p_labels = 1. / (1 + np.exp(-logits))
    labels = np.int32(p_labels > np.random.rand(num_examples, 1))
    labels = np.squeeze(2*labels - 1)
    # labels = labels.astype(int)

    logits_test = np.reshape(np.dot(design_matrix_test, random_weights) + random_bias, (-1, 1))
    p_labels_test = 1. / (1 + np.exp(-logits_test))
    labels_test = np.int32(p_labels_test > np.random.rand(num_examples_test, 1))
    labels_test = np.squeeze(2*labels_test - 1)
    return design_matrix, design_matrix_test, labels, labels_test