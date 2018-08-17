import numpy as np

def pca_preprocess(X, bit):
    '''
    Input:
        X: n * input_dim vectors
        bit: output_dim

    Output:
        W: input_dim * output_dim maxtrix for PCA
        score: value for contribution (>= 0.85 is accepted)
    '''
    row, col = X.shape
    for i in range(col):
        X[:, i] -= X[:, i].mean()
    # each col = dim = variable
    cov = np.cov(X, rowvar=False)

    # eigenvalues and eigenvectors
    val, vec = np.linalg.eig(cov)

    idx = np.argsort(-val)
    sort_val = val[idx]
    sort_vec = vec[:, idx]

    idx = idx[:bit]
    W = sort_vec[:, idx]

    score = 1.0 * sum(sort_val[:bit] / sum(sort_val))
    return W, score

def itq(V, iters = 50):
    '''
    Input:
        V: n * input_dim vectors
        iter: num of iterations

    Output:
        B: input_dim * output_dim maxtrix for PCA
        R: value for contribution (>= 0.85 is accepted)
    '''
    # initialize with a orthogonal random rotation matrix R
    (number, bit) = V.shape

    # Gaussian distribution of mean 0 and variance 1
    R = np.random.randn(bit, bit)
    U, V2, S2 = np.linalg.svd(R)
    R = U[:, range(0, bit)]

    # Fix and Update iterations
    for i in range(iters):
        print 'Iteration ' + str(i + 1) + ' loading..'
        # Fix R and update B(UX)
        Z = V.dot(R)
        (row, col) = Z.shape
        UX = np.ones((row, col)) * -1
        UX[Z >= 0] = 1

        # Fix B and update R
        C = UX.T.dot(V)
        UB, sigma, UA = np.linalg.svd(C)
        UA = UA.T
        R = UA.dot(UB.T)

    B = UX

    # Transform into binary code
    B[B < 0] = 0

    return (B, R)
