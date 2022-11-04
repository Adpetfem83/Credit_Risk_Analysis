# Credit_Risk_Analysis



![lendinglogo](https://github.com/Adpetfem83/Credit_Risk_Analysis/blob/main/Images/LendingClub.png)

Credit Risk Analysis uses Machine Learning algorithms to identify credit card risk using a dataset from LendingClub.

# Overview

The aim and purpose of this challenge is to understand how `Machine Learning` statistical algorithms could be applied to make predictions based on the pattern of the data given. This particular challenge focuses on Supervised Learning(for the data is labeled) and a free dataset from **LendingClub**, a P2P lending service company that analyses and predicts individual's credit risk 

Meanwhile, I applied different `Machine Learning` techniques to train and analyse the data with unbalanced classes. This dataset used has an unbalanced classification isses due to the number of good loans outweighing the amount of risky loans. Therefore, I tried to balance out the classifications in order to allow more meaningful predictions and improve the accuracy score, we then employed various `Machine Learning` algorithms to resample the data. Some of the algorithms used include the following `RandomOverSampler`, `SMOTE`, `ClusterCentroids`, `SMOTEENN`, `BalancedRandomForestClassifier`, and `EasyEnsembleClassifier`. We also used Python'libraries such as 'scikit-learn', 'imbalanced-learn' in order to analyse the results and provide a basis for comparison.

# Results

The Original dataset "LoanStats_2019Q.csv" revealed a total of 115,675 loan applications that were provided. We then used the "loan status" to determine whether the application was considered "low" or "high" risk. The applications that had "current" as the "loan status" were classified as "low risk" and the remaining as "high risk". This ultimately reduced the dataset to 68,817 total applications with 99% classified as "low risk". 

![datacount](https://github.com/Adpetfem83/Credit_Risk_Analysis/blob/main/Images/Figure_1.png)

Using the split method to categorize the data for training vs testing, 51,366 "low risk" and 246 "high risk" applications were categorized into the training set.   

![trainingdata](https://github.com/Adpetfem83/Credit_Risk_Analysis/blob/main/Images/Figure_2.png)

## Deliverable 1: Use Resampling Models to Predict Credit Risk

### Oversampling

**`RandomOverSampler Model`** randomly selects from the minority class and adds it to the training set until both classifications are equal. The results classified 51,366 records each as High Risk and Low Risk.

![oversamplecount](https://github.com/Adpetfem83/Credit_Risk_Analysis/blob/main/Images/Figure_3.png)

  * Balanced accuracy score: 63.6%.

  ![oversampleacc](https://github.com/Adpetfem83/Credit_Risk_Analysis/blob/main/Images/Figure_4.png)

  * The "High Risk" precision rate was only 1% with the recall at 68% giving this model an F1 score of 2%.
  * "Low Risk" had a precision rate of 100% and recall at 59%.  
  
  ![oversamplecm](https://github.com/Adpetfem83/Credit_Risk_Analysis/blob/main/Images/Figure_5a.png)
  
  ![oversampleclass](https://github.com/Adpetfem83/Credit_Risk_Analysis/blob/main/Images/Figure_5b.png)

**`SMOTE (Synthetic Minority Oversampling Technique) Model`**, like `RandomOverSampler` increases the size of the minority class by creating new values based on the value of the closest neighbors to the minority class instead of random selection. 

  * The balanced accuracy score improved slightly to 66.2%.

  ![smoteacc](https://github.com/Adpetfem83/Credit_Risk_Analysis/blob/main/Images/SMOTE_Figure_6.png)

  * Like `RandomOverSampler`, the "High Risk" precision rate again was only 1% with the recall degraded to 62% giving this model an F1 score of 2%.
  * "Low Risk" had a precision rate of 100% and an improved recall at 69%.  

  ![smotecm](https://github.com/Adpetfem83/Credit_Risk_Analysis/blob/main/Images/Figure_7a.png)
  
  ![smoteclass](https://github.com/Adpetfem83/Credit_Risk_Analysis/blob/main/Images/Figure_7b.png)

### Undersampling

**`ClusterCentroids Model`**, an algorithm that identifies clusters of the majority class to generate synthetic data points that are representative of the clusters. The model classified 246 records each as High Risk and Low Risk.

![undersamplecount](https://github.com/Adpetfem83/Credit_Risk_Analysis/blob/main/Images/Figure_8.png)

  * Balanced accuracy score was lower than the oversampling models at 54.5%.

  ![underacc](https://github.com/Adpetfem83/Credit_Risk_Analysis/blob/main/Images/Figure_9.png)

  * The "High Risk" precision rate again was only at 1% with the recall at 69% giving this model an F1 score of 1%.
  * "Low Risk" had a precision rate of 100% and with a lower recall at 40% compared to the oversampling models.  

  ![undercm](https://github.com/Adpetfem83/Credit_Risk_Analysis/blob/main/Images/Figure_10a.png)
  
  ![underclass](https://github.com/Adpetfem83/Credit_Risk_Analysis/blob/main/Images/Figure_10b.png)

## Deliverable 2: Use the SMOTEENN algorithm to Predict Credit Risk

### Combination Sampling

**`SMOTEENN (Synthetic Minority Oversampling Technique + Edited NearestNeighbors) Model`** combines aspects of both oversampling and undersampling. 

![SMOTEENNcount](https://github.com/Adpetfem83/Credit_Risk_Analysis/blob/main/Images/Figure_11.png)

  * The balanced accuracy score improved to 66.2% when using a combined sampling model.

  ![SMOTEENNacc](https://github.com/Adpetfem83/Credit_Risk_Analysis/blob/main/Images/Figure_12.png)

  * The "High Risk" precision rate did not improve appreciably, in fact it increased by only 1%, however the recall increased significantly to 77% giving this model an F1 score of 2%.
  * "Low Risk" still showed a precision rate of 100% with the recall at 55%.  
  
  ![SMOTEENNcm](https://github.com/Adpetfem83/Credit_Risk_Analysis/blob/main/Images/Figure_13a.png)

  ![SMOTEENNclass](https://github.com/Adpetfem83/Credit_Risk_Analysis/blob/main/Images/Figure_13b.png)

## Deliverable 3: Use Ensemble Classifiers to Predict Credit Risk

Compare two new `Machine Learning` models that reduce bias to predict credit risk. The models classified 17,104 as low Risk and 101 as High Risk.

![Balancedcount](https://github.com/Adpetfem83/Credit_Risk_Analysis/blob/main/Images/Figure_14.png)

**`BalancedRandomForestClassifier Model`**, two trees of the same size and equal size to the minority class are constructed to represent one for the majority class and one for the minority class. 

  * The balanced accuracy score increased to 78.8% for this model.

  ![balanceacc](https://github.com/Adpetfem83/Credit_Risk_Analysis/blob/main/Images/Figure_15.png)

  * The "High Risk precision rate increased to 3% with the recall at 70% giving this model an F1 score of 6%.
  * "Low Risk" still had a precision rate of 100% with the recall at 87%.  
  * The top feature by importance was "total_rec_prncp" at 7.9% of the total.

  ![balancecm](https://github.com/Adpetfem83/Credit_Risk_Analysis/blob/main/Images/Figure_16.png)
  
  ![balance_features](https://github.com/Adpetfem83/Credit_Risk_Analysis/blob/main/Images/Figure_17.png)

**`EasyEnsembleClassifier Model`**, a set of classifiers where individual decisions are combined to classify new examples.

  * The balanced accuracy score increased to 93.2% with this model.

  ![easyeacc](https://github.com/Adpetfem83/Credit_Risk_Analysis/blob/main/Images/Figure_18.png)

  * The "High Risk precision rate increased to 9% with the recall at 92% giving this model an F1 score of 16%.
  * "Low Risk" still had a precision rate of 100% with the recall now at 94%.  

  ![easycm](https://github.com/Adpetfem83/Credit_Risk_Analysis/blob/main/Images/Figure_19.png)
  

# Summary

Analysis of all the six models revealed that, `EasyEnsembleClassifer` model produced the best results with an accuracy rate of 93.2% and a 9% precision rate when predicting "High Risk candidates. The sensitivity rate (aka recall) was also the highest at 92% compared to the other models. The result for predicting "Low Risk" was also the highest with the sensitivity rate at 94% and an F1 score of 97%. Therefore, if a model needed to be recommended to perform this type of analysis, then this one would be the clear choice.

**Ranking of models in descending order based on "High Risk" results:**
* `EasyEnsembleClassifer`: 93.2% accuracy, 9% precision, 92% recall, and 16% F1 Score
* `BalancedRandomForestClassifer`: 78.9% accuracy, 3% precision, 70% recall and 6% F1 Score
* `SMOTE`: 66.2% accuracy, 1% precision, 63% recall and 2% F1 Score
* `SMOTEENN`: 66.2% accuracy, 1% precision, 72% recall and 2% F1 Score
* `RandomOverSampler`: 63.6% accuracy, 1% precision, 77% recall and 2% F1 Score
* `ClusterCentroids`: 54.5% accuracy, 1% precision, 69% recall and 1% F1 Score

A side note that should be considered is that original dataset had 99% of the applications classified as "Low Risk" with only 1% of the data classified in the "High Risk" category. This may skew the results greatly as there is a risk that the `Machine Learning` algorithms are creating clusters drawing from too small of a dataset of actual "High Risk" applications. This margin of risk might not be something that banks would be comfortable accepting.

# Resources
                                              
* Dataset from LendingClub: [LoanStats_2019Q](https://github.com/Adpetfem83/Credit_Risk_Analysis/blob/main/LoanStats_2019Q.csv)
* Software: Python 3.7.9, Anaconda 4.9.2 and Jupyter Notebooks 6.1.4
