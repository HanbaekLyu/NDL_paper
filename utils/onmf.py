# (setq python-shell-interpreter "./venv/bin/python")


# import tensorflow as tf
import numpy as np
# import progressbar
# import imageio
import matplotlib.pyplot as plt
from numpy import linalg as LA
import tensorly as tl
from tensorly.tenalg import khatri_rao
from tensorly import unfold as tl_unfold
from tensorly.decomposition import parafac
from sklearn.decomposition import SparseCoder

DEBUG = False


class Online_NMF():
    # Reference: Hanbaek Lyu, Deanna Needell, and Laura Balzano,
    # “Online matrix factorization for markovian data and applications to network dictionary learning.” (2019)
    # https://arxiv.org/abs/1911.01931

    def __init__(self,
                 X, n_components=100,
                 iterations=500,
                 sub_iterations=10,
                 batch_size=20,
                 ini_dict=None,
                 ini_A=None,
                 ini_B=None,
                 ini_C=None,
                 history=0,
                 mode=0,
                 learn_joint_dict=False,
                 alpha=None,
                 beta=None,
                 subsample=True):
        '''
        X: data tensor (3-dimensional) (d x n x m)
        Seeks to find nonnegative matrix factorization X^(2) \approx W^(2) * H
        X^(2) = (dn x m), W^(2) = (dn by r), H = (r by m)
        n_components (int) = r = number of columns in dictionary matrix W where each column represents on topic/feature
        iter (int): number of iterations where each iteration is a call to step(...)
        batch_size (int): number random of columns of X that will be sampled during each iteration
        '''
        self.X = X
        self.n_components = n_components
        self.batch_size = batch_size
        self.iterations = iterations
        self.sub_iterations = sub_iterations
        self.initial_dict = ini_dict
        self.initial_A = ini_A
        self.initial_B = ini_B
        self.initial_C = ini_C
        self.history = history
        self.alpha = alpha
        self.beta = beta
        self.mode = mode  # mode of unfolding the input tensor to learn marginal dictioanry by OMF problem
        self.learn_joint_dict = learn_joint_dict
        self.code = np.zeros(shape=(X.shape[1], n_components))
        self.subsample = subsample

    def unfold_tensor(self, X):
        r = self.n_components
        if not self.learn_joint_dict:
            X_unfold = tl_unfold(X, mode=self.mode)
            # d, n = X_unfold.shape
        else:
            X_unfold = tl_unfold(X, mode=self.mode).T
            # d, n = X_unfold.shape
        return X_unfold

    def sparse_code(self, X, W):
        '''
        Given data matrix X and dictionary matrix W, find
        code matrix H such that W*H approximates X

        args:
            X (numpy array): data matrix with dimensions: features (d) x samples (n)
            W (numpy array): dictionary matrix with dimensions: features (d) x topics (r)

        returns:
            H (numpy array): code matrix with dimensions: topics (r) x samples(n)
        '''

        if DEBUG:
            print('sparse_code')
            print('X.shape:', X.shape)
            print('W.shape:', W.shape, '\n')

        # initialize the SparseCoder with W as its dictionary
        # then find H such that X \approx W*H
        if self.alpha == None:
            coder = SparseCoder(dictionary=W.T, transform_n_nonzero_coefs=None,
                                transform_alpha=2, transform_algorithm='lasso_lars', positive_code=True)
        else:
            coder = SparseCoder(dictionary=W.T, transform_n_nonzero_coefs=None,
                                transform_alpha=self.alpha, transform_algorithm='lasso_lars', positive_code=True)
        # alpha = L1 regularization parameter.
        H = coder.transform(X.T)

        # transpose H before returning to undo the preceding transpose on X
        return H

    def sparse_code_tensor(self, X, W):
        '''
        Given data matrix X and dictionary matrix W, find
        code matrix H such that W*H approximates X

        args:
            X (numpy array): data matrix with dimensions: features (d) x samples (n)
            W (numpy array): dictionary matrix with dimensions: features (d) x topics (r)

        returns:
            H (numpy array): code matrix with dimensions: topics (r) x samples(n)
        '''

        if DEBUG:
            print('sparse_code')
            print('X.shape:', X.shape)
            print('W.shape:', W.shape, '\n')

        # Unfold input tensor and find sparse code
        X_unfold = self.unfold_tensor(X)
        H = self.sparse_code(X_unfold, W)

        return H

    def update_dict(self, W, A, B):
        '''
        Updates dictionary matrix W using new aggregate matrices A and B

        args:
            W (numpy array): dictionary matrix with dimensions: data_dim (d) x topics (r)
            A (numpy array): aggregate matrix with dimensions: topics (r) x topics(r)
            B (numpy array): aggregate matrix with dimensions: topics (r) x data_dim (d)

          returns:
            W1 (numpy array): updated dictionary matrix with dimensions: features (d) x topics (r)
        '''
        # extract matrix dimensions from W
        # and initializes the copy W1 that is updated in subsequent for loop
        d, r = np.shape(W)
        W1 = W.copy()

        # ****
        for j in np.arange(r):
            # W1[:,j] = W1[:,j] - (1/W1[j,j])*(np.dot(W1, A[:,j]) - B.T[:,j])
            W1[:, j] = W1[:, j] - (1 / (A[j, j] + 1)) * (np.dot(W1, A[:, j]) - B.T[:, j])
            W1[:, j] = np.maximum(W1[:, j], np.zeros(shape=(d,)))
            W1[:, j] = (1 / np.maximum(1, LA.norm(W1[:, j]))) * W1[:, j]

        return W1

    def step(self, X, A, B, C, W, t):
        '''
        Performs a single iteration of the online NMF algorithm from
        Han's Markov paper.
        Note: H (numpy array): code matrix with dimensions: topics (r) x samples(n)

        args:
            X (numpy array): data matrix with dimensions: data_dim (d) x samples (n)
            A (numpy array): aggregate matrix with dimensions: topics (r) x topics(r)
            B (numpy array): aggregate matrix with dimensions: topics (r) x data_dim(d)
            W (numpy array): dictionary matrix with dimensions: data_dim (d) x topics (r)
            t (int): current iteration of the online algorithm

        returns:
            Updated versions of H, A, B, and W after one iteration of the online NMF
            algorithm (H1, A1, B1, and W1 respectively)
        '''

        # Compute H1 by sparse coding X using dictionary W
        H1 = self.sparse_code(X, W)

        if DEBUG:
            print(H1.shape)

        # Update aggregate matrices A and B
        t = t.astype(float)
        if self.beta == None:
            beta = 1
        else:
            beta = self.beta
        A1 = (1 - (t ** (-beta))) * A + t ** (-beta) * np.dot(H1.T, H1)
        B1 = (1 - (t ** (-beta))) * B + t ** (-beta) * np.dot(H1.T, X.T)
        C1 = (1 - (t ** (-beta))) * C + t ** (-beta) * np.dot(X, X.T)

        # Update dictionary matrix
        W1 = self.update_dict(W, A, B)
        self.history = t + 1
        # print('history=', self.history)
        return H1, A1, B1, C1, W1

    def train_dict(self):
        '''
        Given data tensor X and mode=i, learn dictionary matrix W and the complementary joint sparse code H.
        Reduce to matrix factorization by unfolding X along mode i

        ---------------if 'learn_joint_dict' = False:
        args:
            X (numpy array): data tensor with dimensions = (d) x (n) x (m)
            W (numpy array): dictionary matrix with dimensions  =  features (d) x topics (r) if mode=0
                                                                =  features (n) x topics (r) if mode=1
                                                                =  features (m) x topics (r) if mode=2

        method:
            X(i) = mode-i (Katri-Rao) unfolding of tensor X with dimensions = (d) x (n m) if mode = 0
                                                                            = (n) x (d m) if mode = 1
                                                                            = (m) x (d n) if mode = 2
            find sparse code H such that X(i) \approx W @ H using online matrix factorization

        returns:
            H (numpy array): code matrix with dimensions    = (r) x (n m) if mode = 0
                                                            = (r) x (d m) if mode = 1
                                                            = (r) x (d n) if mode = 2

                                                            if 'learn_joint_dict = False':

        ---------------if 'learn_joint_dict' = True:
        args:
            X (numpy array): data tensor with dimensions = (d) x (n) x (m)
            W (numpy array): dictionary matrix with dimensions  =  features (n m) x topics (r) if mode=0
                                                                =  features (d m) x topics (r) if mode=1
                                                                =  features (d n) x topics (r) if mode=2

        method:
            X(i) = mode-i (Katri-Rao) unfolding of tensor X with dimensions = (n m) x (d) if mode = 0
                                                                            = (d m) x (n) if mode = 1
                                                                            = (d n) x (m) if mode = 2
            find sparse code H such that X(i) \approx W @ H using online matrix factorization

        returns:
            H (numpy array): code matrix with dimensions    = (r) x (d) if mode = 0
                                                            = (r) x (n) if mode = 1
                                                            = (r) x (m) if mode = 2
        '''

        r = self.n_components

        X_unfold = self.unfold_tensor(self.X)
        d, n = X_unfold.shape

        self.code = np.zeros(shape=(r, n))
        code = self.code

        if self.initial_dict is None:
            # initialize dictionary matrix W with random values
            # and initialize aggregate matrices A, B with zeros
            W = np.random.rand(d, r)
            A = np.zeros((r, r))
            B = np.zeros((r, d))
            C = np.zeros((d, d))
            t0 = self.history
        else:
            W = self.initial_dict
            A = self.initial_A
            B = self.initial_B
            C = self.initial_C
            t0 = self.history

        for i in np.arange(1, self.iterations):
            idx = np.arange(n)
            # randomly choose batch_size number of columns to sample
            # initializing the "batch" of X, which are the subset
            # of columns from X_unfold that were randomly chosen above
            if self.subsample:
                idx = np.random.randint(n, size=self.batch_size)

            X_batch = X_unfold[:, idx]
            # iteratively update W using batches of X, along with
            # iteratively updated values of A and B
            # print('X.shape before training step', self.X.shape)
            H, A, B, C, W = self.step(X_batch, A, B, C, W, t0 + i)
            code[:, idx] += H.T
            # print('dictionary=', W)
            # print('code=', H)
            # plt.matshow(H)

            #  progress status
            # print('Current iteration %i out of %i' % (i, self.iterations))
        return W, A, B, C, code

def update_code_within_radius(X, W, H0=None, r=None, alpha=0, sub_iter=10, stopping_diff=0.1):
    '''
    Find \hat{H} = argmin_H ( | X - WH| + alpha|H| ) within radius r from H0
    Use row-wise projected gradient descent
    Do NOT sparsecode the whole thing and then project -- instable
    12/5/2020 Lyu
    Using sklearn's SparseCoder for ALS seems unstable (errors could increase due to initial overshooting and projection)
    '''

    A = W.T @ W
    B = W.T @ X

    if H0 is None:
        H0 = np.random.rand(W.shape[1], X.shape[1])

    H1 = H0.copy()

    i = 0
    dist = 1
    while (i<sub_iter) and (dist>stopping_diff):
        H1_old = H1.copy()
        for k in np.arange(H1.shape[0]):
            grad = (np.dot(A[k,:], H1) - B[k,:]+alpha*np.ones(H1.shape[1]))
            # H1[k, :] = H1[k,:] - (1 / (A[k, k] + np.linalg.norm(grad, 2))) * grad
            H1[k, :] = H1[k,:] - (1 / ( ((i+10)**(0.5))* (A[k, k] + 1))) * grad
            # use i+10 to ensure monotonicity (but gets slower)
            H1[k,:] = np.maximum(H1[k,:], np.zeros(shape=(H1.shape[1],)))  # nonnegativity constraint
            if r is not None:  # usual sparse coding without radius restriction
                d = np.linalg.norm(H1 - H0, 2)
                H1 = H0 + (r/max(r, d))*(H1 - H0)
            H0 = H1

        dist = np.linalg.norm(H1 - H1_old, 2)/np.linalg.norm(H1_old, 2)
        # print('!!! dist', dist)
        H1_old = H1
        i = i+1
        # print('!!!! i', i)  # mostly the loop finishes at i=1 except the first round

    return H1
