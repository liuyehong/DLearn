# DLearn
Delaunay Triangulation Learner and Random Crystal

## Description
The Delaunay triangulation learner (DTL) is a new statistical learner that is constructed based on data Delaunay triangulation. The package contains modules including bagging DTL and random crystal, which can solve both regression and classification problems.

## Prerequisites
What things you need to install the software and how to install them
```
Scipy package
Numpy package
Scikit-learn package
```

## Instructions
### Bagging DTL
```
DBaggingRegressor(n_estimator, max_dim, n_bootstrap, weight_inside, greedy)
```
The DBaggingRegressor is a bagging Delaunay triangulation learner for regression problems.

#### Attributes
* n_estimator: The total number of ensembled base learners, which has the default value of 100. 
* max_dim: The maximum number of dimensions. The default value is the dimension of the feature space.
* n_bootstrap: The proportion of the bootstrapped samples, which has the default value of 0.9.
* weight_inside: The weight assigned to the base learners that contain the point of prediction inside their convex hulls, which has the default value of 0.99.
* greedy: The parameter controlling whether to use the marginal greedy search method, which has the default value as `greedy'.
* subspace_importance: The frequency of a subspace that has been used as the optimal subspace to build a DTL. 

#### Methods
* fit(X_train, Y_train): The model fitting method.
* predict(X_predict): The model prediction method.
* mse(X_test, Y_test): The mean squared error evaluated on the test data.
* var_imp(X_train, Y_train): The function of variable importance.

```
DBaggingClassifier(n_estimator, max_dim, n_bootstrap, weight_inside, greedy)
```
The DBaggingClassifier is a bagging Delaunay triangulation learner for classification problems.

#### Attributes
* n_estimator: The total number of ensembled base learners, which has the default value of 100. 
* max_dim: The maximum number of dimensions. The default value is the dimension of the feature space.
* n_bootstrap: The proportion of the bootstrapped samples, which has the default value of 0.9.
* weight_inside: The weight assigned to the base learners that contain the point of prediction inside their convex hulls, which has the default value of 0.99.
* greedy: The parameter controlling whether to use the marginal greedy search method, which has the default value as `greedy'.
* subspace_importance: The frequency of a subspace that has been used as the optimal subspace to build a DTL. 

#### Methods
* fit(X_train, Y_train): The model fitting method.
* predict(X_predict): The model prediction method.
* mcr(X_test, Y_test): The misclassification rate evaluated on the test data.
* var_imp(X_train, Y_train): The function of variable importance.


## Random crystal
```
RandomCrystalRegressor(n_estimator, max_dim, n_bootstrap, p_bootstrap, weight_inside)
```
The RandomCrystalRegressor is a random crystal for regression problems.

#### Attributes
* n_estimator: The total number of ensembled base learners, which has the default value of 100. 
* max_dim: The maximum number of dimensions. The default value is the dimension of the feature space.
* n_bootstrap: The proportion of the bootstrapped samples, which has the default value of 0.9.
* p_bootstrap: The proportion of the bootstrapped subspaces, which has the default value of 1.
* weight_inside: The weight assigned to the base learners that contain the point of prediction inside their convex hulls, which has the default value of 0.99.
* subspace_importance: The frequency of a subspace that has been used as the optimal subspace to build a DTL. 

#### Methods
* fit(X_train, Y_train): The model fitting method.
* predict(X_predict): The model prediction method.
* mse(X_test, Y_test): The mean squared error evaluated on the test data.
* var_imp(X_train, Y_train): The function of variable importance.

```
RandomCrystaClassifier(n_estimator, max_dim, n_bootstrap, p_bootstrap, weight_inside)
```
The RandomCrystaClassifier is a random crystal for classification problems.

#### Attributes
* n_estimator: The total number of ensembled base learners, which has the default value of 100. 
* max_dim: The maximum number of dimensions. The default value is the dimension of the feature space.
* n_bootstrap: The proportion of the bootstrapped samples, which has the default value of 0.9.
* p_bootstrap: The proportion of the bootstrapped subspaces, which has the default value of 1.
* weight_inside: The weight assigned to the base learners that contain the point of prediction inside their convex hulls, which has the default value of 0.99.
* subspace_importance: The frequency of a subspace that has been used as the optimal subspace to build a DTL. 

#### Methods
* fit(X_train, Y_train): The model fitting method.
* predict(X_predict): The model prediction method.
* mcr(X_test, Y_test): The misclassification rate evaluated on the test data.
* var_imp(X_train, Y_train): The function of variable importance.

## Example
### Simulation
```
from DBaggingRegressor import *
import numpy as np

n_train = 100  # sample size
n_test = 100
p = 5  # dimension of features

# Generate data from a sparse linear model
def f(X, std):
    return np.sum(X[:, :2], axis=1) + std*np.random.normal(size=len(X))
    
def data_generator(f, n, p):  #sample size = n and dimension = p
    X_train = np.random.normal(scale=1, size=[n, p])
    Y_train = f(X_train, std=0.1)
    return X_train, Y_train

# Simulation
for i in range(100):
    X_train, Y_train = data_generator(f, n_train, p)  # generate training data from f(X)
    X_test, Y_test = data_generator(f, n_test, p)  # generate testing data from f(X)
    
    dbag = DBaggingRegressor(n_estimator=100, n_bootstrap=0.9, greedy='no', max_dim=2)
    dbag.fit(X_train, Y_train)
    mse = dbag.mse(X_test, Y_test)
    List_dbag.append(mse)
    
print np.average(List_dbag)
```
### Results
```
0.0671
```

## Authors

* **Yehong Liu (liuyh@hku.hk) and Guosheng Yin (gyin@hku.hk)**



