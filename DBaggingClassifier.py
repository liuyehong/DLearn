from F_D_Lambda_Classifier import *
import itertools
from scipy.spatial import ConvexHull
import warnings


class DBagging_Classifier():
    def __init__(self, n_estimator=None, max_dim=None, n_bootstrap=None,
                 Lambda=None, alpha=None, eps=None, h=None,
                 List_f_d_lambda=None, List_is_in=None, List_subspace=None, initial=None,
                 weight_inside=None, max_features=None, greedy=None):

        self.variable_importance = []
        self.max_features = max_features
        self.n_estimator = n_estimator
        self.max_dim = max_dim
        self.n_bootstrap = n_bootstrap
        self.Lambda = Lambda
        self.alpha = alpha
        self.eps = eps
        self.h = h
        self.List_f_d_lambda = List_f_d_lambda
        self.List_is_in = List_is_in
        self.List_subspace = List_subspace
        self.initial = initial
        self.weight_inside = weight_inside
        self.greedy = greedy


        if self.weight_inside is None:
            self.weight_inside = 0.99

        if self.List_f_d_lambda is None:
            self.List_f_d_lambda = []

        if self.List_is_in is None:
            self.List_is_in = []

        if self.n_estimator is None:
            self.n_estimator = 100

        if self.max_dim is None:
            self.max_dim = 2

        if self.initial is None:
            self.initial = 'Y'
        if self.Lambda is None:
            self.Lambda = 0


    def fit(self, X_train, Y_train):
        if self.greedy is None:
            self.greedy = 'greedy'

        self.List_subspace = []

        if self.max_dim is None:
            self.max_dim = np.size(X_train, axis=1)

        n = len(Y_train)
        if self.n_bootstrap is None:
            self.n_bootstrap = 0.9

        def Bootstrap(X, Y):
            sample_size = len(X)
            bootstrap_idx = np.random.choice(range(sample_size), size=int(n*self.n_bootstrap), replace=False)
            oob_idx = list(set(range(len(Y)))-set(bootstrap_idx))

            Xb = X[bootstrap_idx, :]
            Yb = Y[bootstrap_idx]
            Xoob = X[oob_idx, :]
            Yoob = Y[oob_idx]
            return Xb, Yb, Xoob, Yoob


        self.List_f_d_lambda = []
        list_dims = []
        for dim_subspace in range(1, self.max_dim + 1):
            list_dims.extend(np.random.permutation(list(itertools.combinations(range(np.size(X_train, axis=1)),
                                                                               dim_subspace))))
        for k in range(self.n_estimator):
            print k

            if self.greedy == 'greedy':
                Xb, Yb, Xoob, Yoob = Bootstrap(X_train, Y_train)  # bootstrap data
                # iterate all possible subspaces and greedy find the optimal one.
                min_mcr = np.inf  # training r square
                # opt_f_d_lambda = F_D_Lambda(Lambda=self.Lambda, alpha=self.alpha, eps=self.eps, h=self.h)
                opt_f_d_lambda = F_D_Lambda_classifier()
                P_set = []
                for d in range(1, np.size(X_train, axis=1) + 1):
                    List_mcr_at_d = []
                    List_subspace_at_d = []
                    List_f_d_lambda = []

                    for i_d in list(set(range(np.size(X_train, axis=1))) - set(P_set)):
                        dims = np.append(P_set, i_d).astype(int)
                        # d=1
                        if len(dims) == 1:
                            try:
                                f_d_lambda = F_D_Lambda_classifier(Lambda=self.Lambda, alpha=self.alpha, eps=self.eps, h=self.h)

                                f_d_lambda.quick_fit1(Xb[:, dims[0]], Yb)
                                mcr = f_d_lambda.mcr(Xoob[:, dims], Yoob)
                                List_mcr_at_d.append(mcr)
                                List_subspace_at_d.append(list(dims))
                                List_f_d_lambda.append(f_d_lambda)
                            except:
                                pass
                        else:
                            # d>=2
                            try:

                                f_d_lambda = F_D_Lambda_classifier(Lambda=self.Lambda, alpha=self.alpha, eps=self.eps, h=self.h,
                                                                   )
                                f_d_lambda.quick_fit(Xb[:, dims], Yb)
                                mcr = f_d_lambda.mcr(Xoob[:, dims], Yoob)

                                # filter the optimal subspace and base learner

                                List_mcr_at_d.append(mcr)
                                List_subspace_at_d.append(list(dims))
                                List_f_d_lambda.append(f_d_lambda)
                            except:
                                pass

                    if len(List_mcr_at_d) == 0:
                        List_mcr_at_d = [np.inf]
                    if np.max(List_mcr_at_d) > min_mcr:
                        break
                    else:
                        min_mcr = np.min(List_mcr_at_d)
                        opt_subspace = List_subspace_at_d[np.argmin(List_mcr_at_d)]
                        opt_f_d_lambda = List_f_d_lambda[np.argmin(List_mcr_at_d)]
                        P_set = opt_subspace

                self.List_subspace.append(opt_subspace)
                self.List_f_d_lambda.append(opt_f_d_lambda)

            else:
                Xb, Yb, Xoob, Yoob = Bootstrap(X_train, Y_train)  # bootstrap data
                # iterate all possible subspaces and greedy find the optimal one.
                min_mcr = np.inf  # training r square
                #opt_f_d_lambda = F_D_Lambda(Lambda=self.Lambda, alpha=self.alpha, eps=self.eps, h=self.h)
                opt_f_d_lambda = F_D_Lambda_classifier()

                if self.max_features == 'sqrt':
                    list_dims_loop = list_dims[:int(np.sqrt(len(list_dims)))]
                elif self.max_features == 'log2':
                    list_dims_loop = list_dims[:int(np.log2(len(list_dims)))]
                else:
                    list_dims_loop = list_dims[:len(list_dims)]

                for dims in list_dims_loop:
                    # d=1
                    if len(dims) == 1:
                        try:
                            f_d_lambda = F_D_Lambda_classifier(Lambda=self.Lambda, alpha=self.alpha, eps=self.eps, h=self.h)

                            f_d_lambda.quick_fit1(Xb[:, dims[0]], Yb)
                            mcr = f_d_lambda.mcr(Xoob[:, dims], Yoob)
                            if mcr < min_mcr:
                                min_mcr = mcr
                                opt_f_d_lambda = f_d_lambda
                                opt_subspace = list(dims)
                        except:
                            pass
                    else:
                        # d>=2
                        try:

                            f_d_lambda = F_D_Lambda_classifier(Lambda=self.Lambda, alpha=self.alpha, eps=self.eps, h=self.h,
                                                    )
                            f_d_lambda.quick_fit(Xb[:, dims], Yb)
                            mcr = f_d_lambda.mcr(Xoob[:, dims], Yoob)

                            # filter the optimal subspace and base learner
                            if mcr < min_mcr:
                                min_mcr = mcr
                                opt_f_d_lambda = f_d_lambda
                                opt_subspace = list(dims)
                        except:
                            #print 'Collinearity'
                            pass

                self.List_subspace.append(opt_subspace)
                self.List_f_d_lambda.append(opt_f_d_lambda)

        return self.List_f_d_lambda, self.List_subspace

    def predict(self, X_predict):
        List_Y_predict = np.zeros([self.n_estimator, len(X_predict)])
        List_is_in = np.zeros([self.n_estimator, len(X_predict)])
        for k in range(self.n_estimator):
            subspace = np.array(self.List_subspace[k])
            f_d_lambda = self.List_f_d_lambda[k]
            f_d_lambda.is_inside(X_predict[:, subspace.astype(int)])
            List_is_in[k, :] = f_d_lambda.is_inside(X_predict[:, subspace.astype(int)])
            Y_predict = f_d_lambda.predict_prob(X_predict[:, subspace.astype(int)])
            List_Y_predict[k, :] = Y_predict

        Para_List_is_in = (1-List_is_in)*self.weight_inside + List_is_in*(1-self.weight_inside)
        Para_List_is_in = Para_List_is_in/np.sum(Para_List_is_in, axis=0)
        List_final_predict = np.zeros(len(X_predict))
        for i in range(np.size(X_predict, 0)):
            List_final_predict[i] =\
                np.dot(List_Y_predict[:, i].transpose(), Para_List_is_in[:, i])
        return List_final_predict > 0.5

    def mcr(self, X_test, Y_test):
        Y_predict = self.predict(X_test)
        MCR = np.average(Y_predict != Y_test)
        return MCR

    def var_imp(self, X_train, Y_train):
        Matrix_mcr = np.zeros([self.n_estimator, 1 + np.size(X_train, axis=1)])

        self.variable_importance = np.zeros(np.size(X_train, axis=1))

        if self.greedy is None:
            self.greedy = 'greedy'

        self.List_subspace = []

        if self.max_dim is None:
            self.max_dim = np.size(X_train, axis=1)

        n = len(Y_train)
        if self.n_bootstrap is None:
            self.n_bootstrap = 0.9

        def Bootstrap(X, Y):
            sample_size = len(X)
            bootstrap_idx = np.random.choice(range(sample_size), size=int(n * self.n_bootstrap), replace=False)
            oob_idx = list(set(range(len(Y))) - set(bootstrap_idx))

            Xb = X[bootstrap_idx, :]
            Yb = Y[bootstrap_idx]
            Xoob = X[oob_idx, :]
            Yoob = Y[oob_idx]
            return Xb, Yb, Xoob, Yoob

        self.List_f_d_lambda = []
        list_dims = []
        for dim_subspace in range(1, self.max_dim + 1):
            list_dims.extend(np.random.permutation(list(itertools.combinations(range(np.size(X_train, axis=1)),
                                                                               dim_subspace))))
        for k in range(self.n_estimator):
            print k

            for m in range(np.size(X_train, axis=1) + 1):
                if self.greedy == 'greedy':
                    if m < np.size(X_train, axis=1):
                        # permute X_m
                        X_train0 = X_train.copy()
                        X_train0[:, m] = np.random.permutation(X_train[:, m])
                        Xb, Yb, Xoob, Yoob = Bootstrap(X_train0, Y_train)  # bootstrap data
                    else:
                        Xb, Yb, Xoob, Yoob = Bootstrap(X_train, Y_train)  # bootstrap data

                    # iterate all possible subspaces and greedy find the optimal one.
                    min_mcr = np.inf  # training r square
                    # opt_f_d_lambda = F_D_Lambda(Lambda=self.Lambda, alpha=self.alpha, eps=self.eps, h=self.h)
                    opt_f_d_lambda = F_D_Lambda_classifier()
                    P_set = []
                    for d in range(1, np.size(X_train, axis=1) + 1):
                        List_mcr_at_d = []
                        List_subspace_at_d = []
                        List_f_d_lambda = []

                        for i_d in list(set(range(np.size(X_train, axis=1))) - set(P_set)):
                            dims = np.append(P_set, i_d).astype(int)
                            # d=1
                            if len(dims) == 1:
                                try:
                                    f_d_lambda = F_D_Lambda_classifier(Lambda=self.Lambda, alpha=self.alpha, eps=self.eps,
                                                                       h=self.h)

                                    f_d_lambda.quick_fit1(Xb[:, dims[0]], Yb)
                                    mcr = f_d_lambda.mcr(Xoob[:, dims], Yoob)
                                    List_mcr_at_d.append(mcr)
                                    List_subspace_at_d.append(list(dims))
                                    List_f_d_lambda.append(f_d_lambda)
                                except:
                                    pass
                            else:
                                # d>=2
                                try:

                                    f_d_lambda = F_D_Lambda_classifier(Lambda=self.Lambda, alpha=self.alpha, eps=self.eps,
                                                                       h=self.h,
                                                                       )
                                    f_d_lambda.quick_fit(Xb[:, dims], Yb)
                                    mcr = f_d_lambda.mcr(Xoob[:, dims], Yoob)

                                    # filter the optimal subspace and base learner

                                    List_mcr_at_d.append(mcr)
                                    List_subspace_at_d.append(list(dims))
                                    List_f_d_lambda.append(f_d_lambda)
                                except:
                                    pass

                        if len(List_mcr_at_d) == 0:
                            List_mcr_at_d = [np.inf]
                        if np.max(List_mcr_at_d) > min_mcr:
                            break
                        else:
                            min_mcr = np.min(List_mcr_at_d)
                            opt_subspace = List_subspace_at_d[np.argmin(List_mcr_at_d)]
                            opt_f_d_lambda = List_f_d_lambda[np.argmin(List_mcr_at_d)]
                            P_set = opt_subspace

                    self.List_subspace.append(opt_subspace)
                    Matrix_mcr[k, m] = min_mcr
                    self.List_f_d_lambda.append(opt_f_d_lambda)

                else:
                    if m < np.size(X_train, axis=1):
                        # permute X_m
                        X_train0 = X_train.copy()
                        X_train0[:, m] = np.random.permutation(X_train[:, m])
                        Xb, Yb, Xoob, Yoob = Bootstrap(X_train0, Y_train)  # bootstrap data

                    else:
                        Xb, Yb, Xoob, Yoob = Bootstrap(X_train, Y_train)  # bootstrap data
                    # iterate all possible subspaces and greedy find the optimal one.
                    min_mcr = np.inf  # training r square
                    # opt_f_d_lambda = F_D_Lambda(Lambda=self.Lambda, alpha=self.alpha, eps=self.eps, h=self.h)
                    opt_f_d_lambda = F_D_Lambda_classifier()

                    if self.max_features == 'sqrt':
                        list_dims_loop = list_dims[:int(np.sqrt(len(list_dims)))]
                    elif self.max_features == 'log2':
                        list_dims_loop = list_dims[:int(np.log2(len(list_dims)))]
                    else:
                        list_dims_loop = list_dims[:len(list_dims)]

                    for dims in list_dims_loop:
                        # d=1
                        if len(dims) == 1:
                            try:
                                f_d_lambda = F_D_Lambda_classifier(Lambda=self.Lambda, alpha=self.alpha, eps=self.eps,
                                                                   h=self.h)

                                f_d_lambda.quick_fit1(Xb[:, dims[0]], Yb)
                                mcr = f_d_lambda.mcr(Xoob[:, dims], Yoob)
                                if mcr < min_mcr:
                                    min_mcr = mcr
                                    opt_f_d_lambda = f_d_lambda
                                    opt_subspace = list(dims)
                            except:
                                pass
                        else:
                            # d>=2
                            try:

                                f_d_lambda = F_D_Lambda_classifier(Lambda=self.Lambda, alpha=self.alpha, eps=self.eps,
                                                                   h=self.h,
                                                                   )
                                f_d_lambda.quick_fit(Xb[:, dims], Yb)
                                mcr = f_d_lambda.mcr(Xoob[:, dims], Yoob)

                                # filter the optimal subspace and base learner
                                if mcr < min_mcr:
                                    min_mcr = mcr
                                    opt_f_d_lambda = f_d_lambda
                                    opt_subspace = list(dims)
                            except:
                                # print 'Collinearity'
                                pass
                    self.List_subspace.append(opt_subspace)
                    Matrix_mcr[k, m] = min_mcr
                    self.List_f_d_lambda.append(opt_f_d_lambda)
        mean_loss_vec = np.average(Matrix_mcr, axis=0)

        VI = [(mean_loss_vec[k] - mean_loss_vec[-1]) for k in range(len(mean_loss_vec) - 1)]
        Var_Importance = np.abs(VI) / np.sum(np.abs(VI))

        return Var_Importance

if __name__ == '__main__':
    from sklearn.datasets import make_moons
    n_train = 100  # sample size
    n_test = 100

    X, Y = make_moons(n_samples=200)
    X_train = X[:100, :]
    Y_train = Y[:100]
    X_test = X[100:, :]
    Y_test = Y[100:]


    db = DBagging_Classifier(n_estimator=10)
    
    db.fit(X_train, Y_train)
    print db.mcr(X_test, Y_test)
    
    print db.var_imp(X_train, Y_train)
