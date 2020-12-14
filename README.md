# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
Dataset contains data about marketing campaigns in banking institution. We seek to predict if campaign was successful, meaning if contacted customer subscribed to product ('yes') or not ('no'). 
Two methods were used - Scikit-learn Logistic Regression, with hyperparameters optimized using HyperDrive and AutoML.

The best performing model - VotingEnsemble, resulting in accuracy of 91.58% was found using AutoML. Best performing parmaeters optimized using HyperDrive resulted in accuracy of 91.10%.

## Scikit-learn Pipeline
Pipeline consisted of computing cluster where Logistic Regression model was trained with hyperparameters tuned with HyperDrive. Hyperdrive optimized parameter C - inverse of regularization strength, and parmater max_iter - maximum number of iterations taken for the solver to converge. Parmeters were sampled randomly resulting in smaller search space than in grid search. 

Additional early stopping Bandit policy was used. The policy helps to optimize time and resources by terminating runs that doesn't fall within the slack factor or slack amount of the evaluation metric with respect to the best performing run.
## AutoML
AutoML object was initialized with "classification" task with metric "accuracy". Seven models were trained before experiment terminated. Best model was VotingEnsemble with accuracy 91.58% followed by MaxAbsScaler XGBoostClassifier 91.51%.

## Pipeline comparison
Accuracy of both methods is comparable with AutoML resulting in slightly better accuracy than LogisticRegression (91.58% vs 91.10%). AutoML run is more robust as it trains many different models, applying various additonal steps like eg. scaling features and seraching for best parameters in one single task, while the same would require many runs with HyperDrive method. 

## Future work
HyperDrive method can be improved by applying wider hyperparameter selection eg. adding regularization and feature scaling. Furthermore dataset is imbalanced and adding class weighting could improve the model.  
AutoML method can be run for longer, resulting in higher number of iterations and possibly even better models.
## Proof of cluster clean up
Cluster deleted in notebook:

![alt text](cluster_delete.png)
