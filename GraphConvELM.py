# -*- coding: utf-8 -*-

import sys

from scipy.linalg import orth
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import accuracy_score
import numpy as np
from numpy import linalg
import scipy.sparse as sp
from scipy.special import expit
from sklearn.metrics import pairwise_kernels, pairwise_distances


class GraphConvELM:

    def __init__(self, n_hidden, activation='sigmoid', reg_para=1., neighbor=10, is_augment=False, seed=None, keep_prob=1.):
        self.n_hidden = n_hidden
        self.reg_para = reg_para
        self.activation = activation
        self.neighbor = neighbor
        self.is_augment = is_augment
        self.keep_prob = keep_prob
        self.seed = seed

    def one2array(self, y, n_dim):
        y_expected = np.zeros((y.shape[0], n_dim))
        for i in range(y.shape[0]):
            y_expected[i][y[i]] = 1
        return y_expected

    def ReLU(self, x):
        return np.maximum(0.0, x)

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)  # only difference

    def sigmoid(self, x):
        return 1. / (1. + np.exp(-x))

    def tanh(self, x):
        return (np.exp(x) - np.exp(-x)) / np.exp(x) + np.exp(-x)

    def drop_edge(self, A, keep_prob=1.):
        # # drop edges
        zero_indx = np.nonzero(A == 0)
        mask = np.random.binomial(n=1, p=keep_prob, size=A.shape)
        A = A * mask
        A[zero_indx] = 0
        A = A + A.T
        A[np.where(A != 2.)] = 0
        A = A/2
        return A

    def __adjacent_mat(self, x, n_neighbors=10):
        """
        Construct normlized adjacent matrix, N.B. consider only connection of k-nearest graph
        :param x: array like: n_sample * n_feature
        :return:
        """
        A = kneighbors_graph(x, n_neighbors=n_neighbors, include_self=True).toarray()
        A = A + A.T
        A = (((A + A.T) > 0) * 1)
        if self.is_augment:
            A = self.drop_edge(A, keep_prob=self.keep_prob)
        # A = A + np.eye(A.shape[0])  # # improve central nodes
        D = np.diag(np.reshape(np.sum(A, axis=1) ** -0.5, -1))
        A = sp.csr_matrix(A)
        D = sp.csr_matrix(D)
        normlized_A = D.dot(A).dot(D)
        return normlized_A

    def fit(self, X_tr, y_tr, X_ts):
        self.sample_weight = None
        if y_tr.shape.__len__() != 2 or y_tr.shape[1] == 1:
            y_tr = y_tr.reshape(-1)
            n_classes_ = np.unique(y_tr).shape[0]
            y_tr = self.one2array(y_tr, n_classes_)
        n_train = X_tr.shape[0]
        X = np.vstack((X_tr, X_ts))
        np.random.seed(self.seed)
        # W = np.random.uniform(-1, 1, size=(X.shape[1], self.n_hidden))
        W = np.random.standard_normal((X.shape[1], self.n_hidden))
        self.W = W
        A_norm = self.__adjacent_mat(X, n_neighbors=self.neighbor)
        if self.activation == 'sigmoid':
            H_ = self.sigmoid(np.dot(A_norm.dot(X), self.W))
        elif self.activation == 'ReLU':
            H_ = self.ReLU(np.dot(A_norm.dot(X), self.W))
        elif self.activation == 'tanh':
            H_ = self.tanh(np.dot(A_norm.dot(X), self.W))
        else:
            raise Exception('Unsupported activation type!')
        self.H_ts = H_[n_train:]
        H_ = A_norm.dot(H_)
        H_tr = H_[:n_train]
        H_ts = H_[n_train:]
        if H_tr.shape[0] >= H_tr.shape[1]:
            TT = np.dot(H_tr.transpose(), H_tr) + self.reg_para * np.eye(H_tr.shape[1])
            inv_ = linalg.inv(TT)
            self.B = np.dot(np.dot(inv_, H_tr.transpose()), y_tr)
        else:
            TT = np.dot(H_tr, H_tr.transpose()) + self.reg_para * np.eye(H_tr.shape[0])
            inv_ = linalg.inv(TT)
            self.B = np.dot(np.dot(H_tr.transpose(), inv_), y_tr)
        y_pre = np.dot(H_ts, self.B)
        self.y_pred = y_pre
        self.label = y_pre.argmax(axis=1)
        return self.label

    def predict(self, X=None):
       return self.label

    def predict_proba(self, X=None):
        return self.y_pred
