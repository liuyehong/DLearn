# DLearn
Delaunay Triangulation Learner and Random Crystal
# Description

The Delaunay triangulation learner (DTL) is a new statistical learner that is constructed based on data Delaunay triangulation. The package contains modules including bagging DTL and random crystal, which can solve both regression and classification problems.

### Prerequisites

What things you need to install the software and how to install them

```
Scipy package
Numpy package
Sklearn package
```
# Instructions
## Bagging DTL
```
DBaggingRegressor(n_estimator, max_dim, n_bootstrap, weight_inside, greedy)
```
The DBaggingRegressor is a bagging Delaunay triangulation learner for regression problems.
### Attributes
* n_estimator: The total number of ensembled base learners, which has the default value as 100. 
* max_dim: The maximum number of dimensions. The default value is the dimension of the dataset.
* n_bootstrap: The proportion of the bootstrapped samples, which has the default value as 0.9.
* weight_inside: The weight assigned to the  base learners that contain the point of prediction inside their convex hulls, which has the default value as 0.99.
* greedy: The parameter controling weather or not to use the marginal greedy search method, which has the default value as 'greedy'.
* subspace_importance: The frequency of a subspace that has been used as the optimal subspace subspace to build a DTL. 
### Methods
* fit(X_train, Y_train): The model fitting method.
* predict(X_predict): The model prediction method.
* score(X_test, Y_test): The score function of R square evaluated based on the test data.
* mse(X_test, Y_test): The mean squared error evaluated based on the test data.
* var_imp(X_train, Y_train): The function of the variable importance.

```
DBaggingClassifier(n_estimator, max_dim, n_bootstrap, weight_inside, greedy)
```
The DBaggingClassifier is a bagging Delaunay triangulation learner for classification problems.
### Attributes
* n_estimator: The total number of ensembled base learners, which has the default value as 100. 
* max_dim: The maximum number of dimensions. The default value is the dimension of the dataset.
* n_bootstrap: The proportion of the bootstrapped samples, which has the default value as 0.9.
* weight_inside: The weight assigned to the  base learners that contain the point of prediction inside their convex hulls, which has the default value as 0.99.
* greedy: The parameter controling weather or not to use the marginal greedy search method, which has the default value as 'greedy'.
* subspace_importance: The frequency of a subspace that has been used as the optimal subspace subspace to build a DTL. 
### Methods
* fit(X_train, Y_train): The model fitting method.
* predict(X_predict): The model prediction method.
* score(X_test, Y_test): The score function of prediction accuracy evaluated based on the test data.
* mcr(X_test, Y_test): The misclassification rate evaluated based on the test data.
* var_imp(X_train, Y_train): The function of the variable importance.


## Random crystal
```
RandomCrystalRegressor(n_estimator, max_dim, n_bootstrap, weight_inside)
```
The RandomCrystalRegressor is a random crystal for regression problems.
### Attributes
* n_estimator: The total number of ensembled base learners, which has the default value as 100. 
* max_dim: The maximum number of dimensions. The default value is the dimension of the dataset.
* n_bootstrap: The proportion of the bootstrapped samples, which has the default value as 0.9.
* p_bootstrap: The proportion of the bootstrapped subspaces, which has the default value as 1.
* weight_inside: The weight assigned to the  base learners that contain the point of prediction inside their convex hulls, which has the default value as 0.99.
* subspace_importance: The frequency of a subspace that has been used as the optimal subspace subspace to build a DTL. 
### Methods
* fit(X_train, Y_train): The model fitting method.
* predict(X_predict): The model prediction method.
* score(X_test, Y_test): The score function of R square evaluated based on the test data.
* mse(X_test, Y_test): The mean squared error evaluated based on the test data.
* var_imp(X_train, Y_train): The function of the variable importance.

```
RandomCrystaClassifier(n_estimator, max_dim, n_bootstrap, weight_inside)
```
The RandomCrystaClassifier is a random crystal for classification problems.
### Attributes
* n_estimator: The total number of ensembled base learners, which has the default value as 100. 
* max_dim: The maximum number of dimensions. The default value is the dimension of the dataset.
* n_bootstrap: The proportion of the bootstrapped samples, which has the default value as 0.9.
* p_bootstrap: The proportion of the bootstrapped subspaces, which has the default value as 1.
* weight_inside: The weight assigned to the  base learners that contain the point of prediction inside their convex hulls, which has the default value as 0.99.
* subspace_importance: The frequency of a subspace that has been used as the optimal subspace subspace to build a DTL. 
### Methods
* fit(X_train, Y_train): The model fitting method.
* predict(X_predict): The model prediction method.
* score(X_test, Y_test): The score function of prediction accuracy evaluated based on the test data.
* mcr(X_test, Y_test): The misclassification rate evaluated based on the test data.
* var_imp(X_train, Y_train): The function of the variable importance.

## Authors

* **Yehong Liu (liuyh@hku.hk) and Guosheng Yin (gyin@hku.hk)**



