from Simplex import *

# Single Delaunay triangulation learner with parameter Z.
class F_D():
    def __init__(self, X_T, Z, tri, List_DXinv):
        self.Z = Z
        self.tri = tri
        self.List_DXinv = List_DXinv
        self.X_T = X_T


    def estimate(self, X_estimate):
        n = len(X_estimate)
        Estimators = np.zeros(n)
        Simplices = self.tri.simplices

        for k in range(n):
            x = X_estimate[k]
            no_simplex_in = self.tri.find_simplex(x)
            DXinv_k = self.List_DXinv[no_simplex_in]

            if no_simplex_in != -1:  # if the point is inside a simplex
                idxes_simplex_in = self.tri.simplices[no_simplex_in]
                S = Simplex(self.X_T[idxes_simplex_in], self.Z[idxes_simplex_in], DXinv_k)
                Estimators[k] = S.estimate_x(x)
            else:
                grav_centers = np.array([np.average(self.tri.points[simplex], axis=0).tolist() for simplex in self.tri.simplices])
                dist = np.sum((x - grav_centers) ** 2, axis=1)
                idx_simplex_nearest = np.argmin(dist)
                Estimators[k] = np.average(self.Z[self.tri.simplices[idx_simplex_nearest]])

        return Estimators

    def is_inside(self, X_estimate):
        n = len(X_estimate)

        IsIn = np.zeros(n)
        for k in range(n):
            x = X_estimate[k, :]
            no_simplex_in = self.tri.find_simplex(x)
            if no_simplex_in != -1:  # if the point is inside a simplex
                IsIn[k] = 0
            else:
                IsIn[k] = 1

        return IsIn



if __name__ == '__main__':
    from DataGenerator import *
    from scipy.spatial import Delaunay
    n_train = 10  # sample size
    n_test = 10
    eps = 0.01  # precision
    alpha = 1  # if the lambda is large, smaller alpha should be used.

    p = 2  # dimension of features
    X_train, Y_train = data_generator(f, n_train, p)  # generate training data from f(X)
    X_test, Y_test = data_generator(f, n_test, p)

    tri = Delaunay(X_train)
    Simplices = tri.simplices
    List_DXinv = []
    for k in range(len(Simplices)):
        X = X_train[Simplices[k]]
        DX = X[0] - X[1:]
        List_DXinv.append(np.linalg.inv(DX))

    fd = F_D(X_T=X_train, Z=Y_train, tri=Delaunay(X_train), List_DXinv=List_DXinv)
    print fd.estimate(X_test)
    #print fd.is_inside(X_test)