# Predictive-Maintenance

The R code involves 3 techniques for predicting the classification of failures and non failures in the data. The following steps have been performed on the dataset:
1. Data cleaning and/or one hot encoding for factor variables.
2. Partitioning data into training and validation.
3. Performing a logistic regression and predicting using the validation dataset. Lift and decile wise charts are constructed for the results obtained from the logistic regression performed.
4. A classification tree has been built on the training dataset, the tree is pruned using the minimum cp value. A confusion matric for the tree has also been provided with an accuracy of 98.2%.
5. A neural network with 1 hidden layer has been fit on the data.
