import numpy as np
from NS_DE_GCELM.GraphConvELM import GraphConvELM
from collections import Counter

class V_GCELM:

    def __init__(self, n_learner=10, n_hidden=10, reg_para=1., n_neighbor=10):
        self.n_learner = n_learner
        self.n_hidden = n_hidden
        self.reg_para = reg_para
        self.n_neighbor = n_neighbor

    def fit(self, X_tr, y_tr, X_ts):
        model_list = []
        y_pre_list = []
        for i in range(self.n_learner):
            model = GraphConvELM(self.n_hidden, activation='sigmoid', reg_para=self.reg_para,
                                 neighbor=self.n_neighbor, is_augment=False, keep_prob=1)
            y_pre_ = model.fit(X_tr, y_tr, X_ts)
            model_list.append(model)
            y_pre_list.append(y_pre_)
        y_pre_list = np.asarray(y_pre_list)  # # n_learner * n_test
        y_pre = np.zeros(y_pre_list.shape[1])
        for i in range(y_pre_list.shape[1]):
            c = Counter(y_pre_list[:, i]).most_common(1)[0][0]
            y_pre[i] = c
        self.y_pre = y_pre
        return y_pre

    def predict(self, X=None):
        return self.y_pre

