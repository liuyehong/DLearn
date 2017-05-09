__author__ = 'CUPL'
import numpy as np

# true model
def f(X, std):

    #return np.sum(X[:, :2], axis=1) + std*np.random.normal(size=len(X))
    return np.sum(np.sin(X[:, :2]), axis=1) + std*np.random.normal(size=len(X))
    #return np.sin(np.sum(X[:, :2], axis=1)) + std*np.random.normal(size=len(X))

    #return np.prod(X[:, :2], axis=1) + std * np.random.normal(size=len(X))
    #return np.prod(np.sin(X[:, :2]), axis=1) + std*np.random.normal(size=len(X))
    #return np.sin(np.prod(X[:, :2], axis=1)) + std*np.random.normal(size=len(X))
    #return np.sin(np.sqrt(np.sum(X[:, :2]**2, axis=1))) + std*np.random.normal(size=len(X))

    #return np.sum(np.cos(X[:, :2]), axis=1) + std*np.random.normal(size=len(X))
# Generate training data
def data_generator(f, n, p):  #sample size = n and dimension = p
    X_train = np.random.normal(scale=1, size=[n, p])
    Y_train = f(X_train, std=0.1)
    return X_train, Y_train




if __name__ == '__main__':
    n_train = 100  # sample size
    n_test = 10
    p = 2  # dimension of features

    X_train, Y_train = data_generator(f, n_train, p)  # generate training data from f(X)
    X_test, Y_test = data_generator(f, n_test, p)  # generate testing data from f(X)




