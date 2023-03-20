## Basic Machine Learning Algorithms in Numpy
This repository contains implementations of basic machine learning algorithms using only the NumPy library in Python.

The following algorithms are included in this repository:

#### Linear Regression
#### Logistic Regression
#### K-Nearest Neighbors (KNN)
#### K-Means


Each of these algorithms is explained in more detail below.


### Linear Regression
Linear regression is a type of regression analysis used to predict the relationship between two continuous variables. In this implementation, we use the ordinary least squares method to find the best fit line for a given set of data points.

The LinearRegression class contains methods for training and testing a linear regression model. The fit method takes in a matrix of input features and a vector of target outputs, and uses the ordinary least squares method to find the coefficients of the linear regression model. The predict method takes in a matrix of input features and returns a vector of predicted target outputs based on the learned coefficients.

### Logistic Regression
Logistic regression is a type of regression analysis used to predict the probability of a binary outcome (i.e., 0 or 1). In this implementation, we use the sigmoid function to map the output of a linear regression model to a value between 0 and 1, which represents the probability of a positive outcome.

The LogisticRegression class contains methods for training and testing a logistic regression model. The fit method takes in a matrix of input features and a vector of binary target outputs, and uses gradient descent to find the coefficients of the logistic regression model. The predict method takes in a matrix of input features and returns a vector of predicted binary target outputs based on the learned coefficients.

### K-Nearest Neighbors (KNN)
K-Nearest Neighbors (KNN) is a type of instance-based learning, where new instances are classified based on the similarity to the k nearest training instances in feature space. In this implementation, we use the Euclidean distance to measure the similarity between instances.

The KNN class contains methods for training and testing a KNN classifier. The fit method takes in a matrix of input features and a vector of target outputs, and simply stores them for later use in classification. The predict method takes in a matrix of input features and returns a vector of predicted target outputs based on the k nearest neighbors in the training set.

### K-Means
K-Means is a type of unsupervised learning used for clustering data points into k clusters. In this implementation, we use the Euclidean distance to measure the similarity between data points and the mean of each cluster as the centroid.

The KMeans class contains methods for clustering a given set of data points. The fit method takes in a matrix of input features and a value of k, and uses the K-Means algorithm to cluster the data points into k clusters. The predict method takes in a matrix of input features and returns a vector of cluster labels for each data point based on the learned centroids.

### Requirements
This implementation requires the NumPy library to be installed in Python.