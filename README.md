# Regression: Simple Linear Regression with California Housing Dataset

In this repository, I explore simple linear regression techniques using the California Housing Dataset.

## Lab Description

This lab exercise focuses on performing single linear regressions with each feature (one column) of the dataset individually. The objective is to calculate two metrics for the predictions obtained from the linear regression: R2 score and Mean Squared Error (MSE).

The process involves:
1. Splitting the dataset into training and testing sets.
2. Selecting one feature at a time.
3. Training the model using that one feature and making predictions.
4. Calculating R2 score and MSE for each separate simple linear regression step.

## Instructions

1. Download the California Housing Dataset.
2. Implement simple linear regression for each feature individually.
3. Print out R2 score and MSE for each feature.
4. Produce a table summarizing the results to determine which feature produced the best results.
5. Write a short explanation of how you would evaluate the results of the simple linear regression estimators, comparing them with the results from multiple linear regression.

## Example Output

Your output should resemble the following:

Multiple Linear Regression using All features
R2 score: 0.6008983115964333, MSE score: 0.5350149774449118

Feature 0: R2 score: 0.4630810035698606, MSE score: 0.7197656965919478
Feature 1: R2 score: 0.013185632224592903, MSE score: 1.3228720450408296
Feature 2: R2 score: 0.024105074271276283, MSE score: 1.3082340086454287
Feature 3: R2 score: -0.0011266270315772875, MSE score: 1.3420583158224824
Feature 4: R2 score: 8.471986797708997e-05, MSE score: 1.3404344471369465
Feature 5: R2 score: -0.00018326453581640756, MSE score: 1.340793693098357
Feature 6: R2 score: 0.020368890210145207, MSE score: 1.3132425427841639
Feature 7: R2 score: 0.0014837207852690382, MSE score: 1.3385590192298276
