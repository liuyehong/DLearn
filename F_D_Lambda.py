from F_D import *
from sklearn.linear_model import LinearRegression
from scipy.spatial import Delaunay
from sklearn.tree import DecisionTreeRegressor
from scipy.interpolate import interp1d
from scipy.optimize import minimize

# Single regularized Delaunay triangulation learner, with methods train and predict.
'''
The F_D_Lambda is the regularized Delaunay triangulation learner.
While training the F_D_Lambda, there are two different initial values
and two different regularization functions.
'''


class F_D_Lambda:
    def __init__(self, X_T=None, Z=None, tri=None, List_DXinv=None, Lambda=None, alpha=None, eps=None, h=None, mode=None,
                 interpol=None, K=None):
        self.Z = Z
        self.tri = tri
        self.List_DXinv = List_DXinv
        self.X_T = X_T
        self.Lambda = Lambda
        self.alpha = alpha
        self.eps = eps
        self.h = h
        self.mode = mode
        self.interpol = interpol
        self.K = K

        if self.mode is None:
            self.mode = 'linear'
        else:
            self.mode = 'tree'


        # Learning rate
        if self.alpha is None:
            self.alpha = 1

        if self.eps is None:
            self.eps = 0.01

        if self.h is None:
            self.h = 10 ** -7

    def ave_total_curvature(self):  # total curvature for each point
        n = len(self.Z)
        Simplices = self.tri.simplices
        p = np.size(Simplices, axis=1)
        n_simplex = np.size(Simplices, axis=0)
        Point_neighbor_normvec = np.zeros([n, n_simplex, p])  # store the norm vector for each simplex
        # seperate normvectors to each point in X_train
        List_start = -np.ones(n)  # list memorize the location for each simplex
        for k in range(len(Simplices)):
            simplex = Simplices[k]
            DXinv_k = self.List_DXinv[k]
            Z_k = self.Z[simplex]
            X_T_k = self.X_T[simplex]
            S = Simplex(X_T_k, Z_k, DXinv_k)
            norm_vec = S.up_norm_vector()
            for idx in simplex:
                List_start[idx] += 1
                Point_neighbor_normvec[idx, int(List_start[idx]), :] = norm_vec
        # compute the regularization function for each point in X_train.
        List_curvature = np.zeros([n, p-1])
        for dim in range(p-1):
            for idx in range(n):
                list_normvec = Point_neighbor_normvec[idx]
                neighbor_normvec = list_normvec[:int(List_start[idx]) + 1, dim]
                Cos_theta = np.dot(neighbor_normvec, neighbor_normvec.transpose())
                if len(neighbor_normvec) > 1:
                    N_N = np.sum(Cos_theta)
                    List_curvature[idx, dim] = (len(neighbor_normvec) ** 2 - N_N) / float(
                        len(neighbor_normvec) * (len(neighbor_normvec) - 1))
                else:
                    List_curvature[idx, dim] = 0

            list_marginal_total_curvature = np.average(List_curvature, axis=0)
        return list_marginal_total_curvature


    def fit(self, X_train, Y_train, initial=None, reg=None):
        if initial is None:
            initial = 'Z0'  # start from linear regression
        if reg is None:
            reg = 'Curvature'

        def LossFun(Z, Y_train):
            return np.var(Z - Y_train)  # square loss

        def RegFun1(X_T, Simplices, Z, List_DXinv):
            n = len(Z)
            p = np.size(Simplices, axis=1)
            n_simplex = np.size(Simplices, axis=0)
            Point_neighbor_normvec = np.zeros([n, n_simplex, p])  # store the norm vector for each simplex
            # seperate normvectors to each point in X_train
            List_start = -np.ones(n)  # list memorize the location for each simplex
            for k in range(len(Simplices)):
                simplex = Simplices[k]
                DXinv_k = List_DXinv[k]
                Z_k = Z[simplex]
                X_T_k = X_T[simplex]
                S = Simplex(X_T_k, Z_k, DXinv_k)
                norm_vec = S.up_norm_vector()
                for idx in simplex:
                    List_start[idx] += 1
                    Point_neighbor_normvec[idx, int(List_start[idx]), :] = norm_vec
            # compute the regularization function for each point in X_train.
            List_curvature = np.zeros(n)
            for idx in range(n):
                list_normvec = Point_neighbor_normvec[idx]
                neighbor_normvec = list_normvec[:int(List_start[idx]) + 1, :]
                Cos_theta = np.dot(neighbor_normvec, neighbor_normvec.transpose())
                if len(neighbor_normvec) > 1:
                    N_N = np.sum(Cos_theta)
                    List_curvature[idx] = (len(neighbor_normvec) ** 2 - N_N) / float(
                        len(neighbor_normvec) * (len(neighbor_normvec) - 1))
                else:
                    List_curvature[idx] = 0
            regularization_fun = np.average(List_curvature)
            return regularization_fun

        def RegFun2(X_T, Simplices, Z, List_DXinv):
            n = len(Z)
            n_simplex = np.size(Simplices, axis=0)
            reg_fun = 0

            for k in range(n_simplex):
                simplex = Simplices[k]
                DXinv_k = List_DXinv[k]
                Z_k = Z[simplex]
                X_T_k = X_T[simplex]
                S = Simplex(X_T_k, Z_k, DXinv_k)
                norm_vec = S.up_norm_vector()
                reg_fun += norm_vec[-1]  # the cos of norm vector and e_z, which is exactly norm_vec[-1]
            return reg_fun/float(n)

        ##
        if reg == 'Curvature':
            RegFun = RegFun1
        else:
            RegFun = RegFun2

        self.X_T = X_train
        self.tri = Delaunay(self.X_T)
        Simplices = self.tri.simplices
        List_DXinv = []

        for k in range(len(Simplices)):
            X = X_train[Simplices[k]]
            DX = X[0] - X[1:]
            List_DXinv.append(np.linalg.inv(DX))
        self.List_DXinv = List_DXinv

        n = len(X_train)
        if initial == 'Y' and self.Lambda == 0:
            self.Z = Y_train
        else:
            if initial == 'Z0':
                # Initialize Z with linear regression
                lngr = LinearRegression()
                lngr.fit(X_train, Y_train)
                self.Z = lngr.predict(X_train)
            else:  # start from Y_train
                self.Z = Y_train

            # Coordinate decent.
            Delta = np.ones(n)
            Lt = LossFun(self.Z, Y_train) + self.Lambda * RegFun(self.X_T, self.tri.simplices, self.Z, self.List_DXinv)

            derta_each_iter = self.eps + 1
            while np.abs(derta_each_iter) > self.eps:
                derta_each_iter = 0  # accumulate improvements for each cycle.
                for k in np.random.permutation(n):
                    d = np.zeros(len(self.Z))
                    d[k] = self.h
                    DeltaLoss_k = (LossFun(self.Z + d, Y_train) - LossFun(self.Z, Y_train)) / self.h
                    DeltaReg_k = (RegFun(self.X_T, self.tri.simplices, self.Z+d, self.List_DXinv)
                                  - RegFun(self.X_T, self.tri.simplices, self.Z, self.List_DXinv)) / self.h
                    Delta[k] = DeltaLoss_k + self.Lambda * DeltaReg_k
                    self.Z[k] -= Delta[k] * self.alpha  # coordinate decent.
                    derta = LossFun(self.Z, Y_train) + \
                            self.Lambda * RegFun(self.X_T, self.tri.simplices, self.Z, self.List_DXinv) - Lt

                    if derta > 0:  # if loss function decay, we take it
                        self.Z[k] += Delta[k] * self.alpha  # coordinate decent.
                    else:
                        Lt += derta
                        derta_each_iter += derta

    def quick_fit(self, X_train, Y_train):

        self.X_T = X_train
        self.tri = Delaunay(self.X_T)
        Simplices = self.tri.simplices
        List_DXinv = []

        for k in range(len(Simplices)):
            X = X_train[Simplices[k]]
            DX = X[0] - X[1:]
            List_DXinv.append(np.linalg.inv(DX))
        self.List_DXinv = List_DXinv
        if self.mode == 'linear':
            lngr = LinearRegression()
            lngr.fit(X_train, Y_train)
            Z0 = lngr.predict(X_train)
        elif self.mode == 'tree':
            dtr = DecisionTreeRegressor()
            dtr.fit(X_train, Y_train)
            Z0 = dtr.predict(X_train)

        self.Z = (Y_train + self.Lambda*Z0)/float(1+self.Lambda)  # This quantity is based on squared regularization.

    # fit for 1d
    def quick_fit1(self, X_train, Y_train):
        self.interpol = interp1d(X_train, Y_train)
        self.x_min = np.min(X_train)
        self.x_max = np.max(X_train)


    def predict(self, X_predict):
        if np.size(X_predict, axis=1) == 1:
            List_predict = []
            for k in range(len(X_predict)):
                if X_predict[k] < self.x_min:
                    List_predict.append(self.interpol(self.x_min))
                elif X_predict[k] > self.x_max:
                    List_predict.append(self.interpol(self.x_max))
                else:
                    List_predict.append(self.interpol(X_predict[k]))
            return List_predict
        else:
            FD = F_D(self.X_T, self.Z, self.tri, self.List_DXinv)
            return FD.estimate(X_predict)


    def score(self, X_test, Y_test):
        Y_predict = self.predict(X_test)
        return 1 - np.var(Y_predict - Y_test) / np.var(Y_test)

    def mse(self, X_test, Y_test):
        Y_predict = self.predict(X_test)
        return np.var(Y_predict - Y_test)

    def is_inside(self, X_predict):
        if np.size(X_predict, axis=1) == 1:
            List_is_inside1 = []
            for k in range(len(X_predict)):
                if X_predict[k] > self.x_min and X_predict[k] < self.x_max:
                    List_is_inside1.append(0)
                else:
                    List_is_inside1.append(1)
            return List_is_inside1
        else:
            FD = F_D(self.X_T, self.Z, self.tri, self.List_DXinv)
            return FD.is_inside(X_predict)



    def tune_lambda(self, X_valid, Y_valid):
        def cv_loss(Lambda):
            Z0 = self.Z
            self.Z = (Y_train + Lambda * Z0) / float(1 + Lambda)
            return np.sum(np.square(Y_valid - self.predict(X_valid)))
        list_cv_loss = [cv_loss(2**j) for j in range(10)]
        j_opt = np.argmin(list_cv_loss)
        Lambda_opt = 2**j_opt
        print j_opt
        print list_cv_loss
        return Lambda_opt



if __name__ == '__main__':
    from DataGenerator import *

    n_train = 100  # sample size
    n_test = 100
    eps = 0.01  # precision
    alpha = 1  # if the lambda is large, smaller alpha should be used.

    p = 2  # dimension of features
    X_train, Y_train = data_generator(f, n_train, p)  # generate training data from f(X)
    X_test, Y_test = data_generator(f, n_test, p)

    fdl = F_D_Lambda(Lambda=100, alpha=1)
    fdl.quick_fit(X_train, Y_train)
    #print fdl.ave_total_curvature()
    #print fdl.predict(X_test)
    fdl.tune_lambda(X_test, Y_test)