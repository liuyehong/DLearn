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

## Examples

### Prediction
```
from DBaggingRegressor import *
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
    
X_train, Y_train = data_generator(f, n_train, p)  # generate training data from f(X)
X_test, Y_test = data_generator(f, n_test, p)  # generate testing data from f(X)
dbag = DBaggingRegressor(n_estimator=100, n_bootstrap=0.9, greedy='no', max_dim=2)
dbag.fit(X_train, Y_train)
print dbag.predict(X_test)
```

### Result
```
[-0.63332027  0.33803703  0.70171951 -0.7277317  -0.29389851  1.49451073
  0.22086775 -0.42406454 -1.93928916  1.00938537 -1.89133195  0.93997194
  0.25927864 -1.19639309 -1.65462006  1.49593518  2.81300328 -0.49088468
 -0.46485142  1.0638482   0.10853272  0.55227528 -0.47504233  0.31277654
 -0.73206829 -0.55696616 -0.56312701  1.6940066   1.75156424  0.67483036
  1.50912366  0.47508003  0.75083365 -0.00897398  1.67920607 -0.50659088
 -0.85079313 -0.20923068 -0.33871617 -0.35256097 -2.87878914  0.19517858
 -2.24689151  0.20353313  0.4041237  -0.02293355  0.31188312 -1.66724448
 -0.87427485 -0.10305542  0.51863488  0.24821254 -0.38311671 -0.88928651
  1.91149502 -1.77621651 -1.37545913  0.37339049  1.82522683 -0.64586055
  2.07632701 -1.54561469  1.51416461 -1.33523164  1.81760813  0.82552768
 -0.23139837  0.12347116  0.46254937  0.27944276 -0.23656042  1.2291315
  1.42560402  1.1478001  -0.95574241  0.9690149   1.00040455 -0.59316946
 -0.14787578  1.30352156  0.37612687  1.53885469  0.8601821  -0.09262812
  1.05977105  0.7946149  -1.33427235 -1.35936915  0.53891978 -0.12317036
  0.21966075  1.37473857  0.30274678 -1.30670981 -1.5796981   1.03846619
 -0.20840201  1.02853775  1.10220153  1.66377596]
```

### Variable importance
```
from DBaggingRegressor import *
n_train = 100  # sample size
p = 5  # dimension of features

# Generate data from a sparse linear model
def f(X, std):
    return np.sum(X[:, :2], axis=1) + std*np.random.normal(size=len(X))
    
def data_generator(f, n, p):  #sample size = n and dimension = p
    X_train = np.random.normal(scale=1, size=[n, p])
    Y_train = f(X_train, std=0.1)
    return X_train, Y_train
    
X_train, Y_train = data_generator(f, n_train, p)  # generate training data from f(X)
dbag = DBaggingRegressor(n_estimator=100, n_bootstrap=0.9, greedy='no', max_dim=2)
print dbag.var_imp(X_train, Y_train)
```

### Result
```
0.0014
```

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

### Result
```
0.0671
```

## Authors

* **Yehong Liu (liuyh@hku.hk) and Guosheng Yin (gyin@hku.hk)**



