__author__ = 'CUPL'
import numpy as np

class Simplex():
    def __init__(self, X_T, Z, DXinv):
        self.X_T = X_T
        self.Z = Z
        self.DXinv = DXinv

    # estimate value for x
    def estimate_x(self, x):
        DZ = self.Z[0] - self.Z[1:]
        G = np.dot(self.DXinv, DZ)  # gradient of simplex
        return np.dot(x - self.X_T[0], G) + self.Z[0]

    # output the up-norm vector of simplex
    def up_norm_vector(self):
        DZ = self.Z[0] - self.Z[1:]
        normvec = np.append(-np.dot(self.DXinv, DZ), 1)
        std_normvec = normvec / np.sqrt(np.sum(normvec ** 2))
        return std_normvec

    def center_estimate(self):
        return np.average(self.Z)


if __name__ == '__main__':
    X = np.array([[0,  0], [1,  0], [0,  1]])
    Z = np.array([1,  0,  0])
    x = np.array([[0, 0], [1, 0], [0, 1]])  # array, n by p.
    DX = X[0]-X[1:]
    DZ = Z[0]-Z[1:]
    DXinv = np.linalg.inv(DX)

    S = Simplex(X_T=X, Z=Z, DXinv=DXinv)
    print S.estimate_x(x)
    print S.up_norm_vector()

