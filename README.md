# Credit_Risk_Analysis

# Overview
A peer to peer lending services company wants to use machine learning to predict credit risk. The company believes this will provide a quicker and more reliable
loan service, and that machine learning will lead to a more accurate prediction of good qualifiers for loans. You are asked to assist the lead data scientist in 
implementing this plan.

## Purpose
Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. We must employ different techniques to train and 
evaluate models with unbalanced classes. We use imbalanced-learn and scikit-learn libraries to build and evaluate models using resampling. Using the credit card 
credit dataset from LendingClub, we oversample the data using the RandomOverSampler and SMOTE algorithms, and undersample the data using the ClusterCentroids 
algorithm. Then, we use a combinatorial approach of over- and undersampling using the SMOTEENN algorithm. Finally, we compare two machine learning models 
that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk. After performing all analysis, we evaluate the performance 
of these models and make a recommendation on whether they should be used to predict credit risk.

# Results

* RandomOverSampler Model

First we use the RandomOverSampler. In random oversampling, instances of the minority class are randomly selected and added to the training set 
until the majority and minority classes are balanced. The model was about 62% accurate and the precision of predicting high risk candidates is much lower
than precision of low risk candidates. The sensitivity between high risk and low risk is much closer.

![oversampler accuracy](https://user-images.githubusercontent.com/99205688/174492847-bce6dc00-0c99-4a35-9172-3ff34d31ba57.PNG)

![oversampler report](https://user-images.githubusercontent.com/99205688/174492849-f73ef4d1-f45a-4e50-b7ac-552969868d5a.PNG)

* SMOTE Model

Next we use SMOTE to oversample, which increases the minority class by interpolating new instances instead of randomly selecting them. The model was about 65% 
accurate with similar precision but more closely spaced sensitivity between high risk and low risk predictions.

![smote accuracy](https://user-images.githubusercontent.com/99205688/174492852-4d22f01c-8e3d-4de7-834f-d577e82ef83b.PNG)

![smote report](https://user-images.githubusercontent.com/99205688/174492854-c349a132-f26f-4028-80db-5663aaba514b.PNG)

* ClusterCentroids Model

Next we perform undersampling, which undersamples the majority class down to the size of the minority class. The ClusteredCentroids model generates synthetic data points that are representative of the clusters.  The model was about 62% accurate and the precision of predicting high risk candidates is much lower
than precision of low risk candidates. The sensitivity to low risk candidates is lower compared to the two previous models.

![clustered accuracy](https://user-images.githubusercontent.com/99205688/174492899-2a3690a9-3d45-4f2d-8c32-c263bb3a49e6.PNG)

![clustered report](https://user-images.githubusercontent.com/99205688/174492904-12526e8f-0ee3-416c-83d4-af307f351e3c.PNG)

* SMOTEENN Model

The SMOTEENN models combines oversampling and undersampling. SMOTEENN oversamples the minority class with SMOTE and cleans the resulting data with an 
undersampling strategy. The model was about 64% accurate (higher than previous models) and the precision was similar to previous models. Sensitivity to high
risk was higher, but low risk sensitivity was still a bit lower than other models.

![smoteen accuracy](https://user-images.githubusercontent.com/99205688/174492910-07cc7a63-815b-4823-92f9-f6ed6fdf3b68.PNG)

![smoteen report](https://user-images.githubusercontent.com/99205688/174492911-1fb6a616-2998-430e-9926-3c9f3886ceb9.PNG)

* BalancedRandomForestClassifier Model

The BalancedRandomForestClassifier randomly under-samples each boostrap sample to balance it. The model was about 65% accurate (higher than previous models) and the precision was much higher for high risk than previous models. While very sensitive to low risk, it was not as sensitive to high risk as previous models.

![balanced forest accuracy](https://user-images.githubusercontent.com/99205688/174492920-a8ea0534-338c-40ee-a31b-5549c4695e60.PNG)

![balanced forest report](https://user-images.githubusercontent.com/99205688/174492926-28ec9a26-2ef4-4198-b97c-031f64e1f0bf.PNG)

* EasyEnsembleClassifier Model

The EasyEnsembleClassifier is an ensemble of AdaBoost learners trained on different balanced boostrap samples. The balancing is achieved by random under-sampling.
The model was about 93% accurate, much higher than previous models. and the precision was much higher for high risk than previous models. The precision for both
high risk and low risk was better than most of the previous models, and the sensitivity for both high and low risk was much higher than previous models.

![easy ensemble accuracy](https://user-images.githubusercontent.com/99205688/174492965-adb6b862-3704-459e-8699-ac37c05af7ed.PNG)

![easy ensemble report](https://user-images.githubusercontent.com/99205688/174492991-d1538315-8851-4c6e-a873-db23b3dcf438.PNG)

# Summary
Given that the dataset count had many more low risk candidates than high risk candidates, we can expect that not all models will be equally good at predicting
those at high risk of credit default. It was useful to undersample and oversample the data through these different models to test which method may be the best
for the company to use moving forward.

## Recommendation 
The EasyEnsembleClassifier Model had the far highest accuracy score out of all the models and I would recommend moving forward with this model. While the precision
at predicting high and low risk was similar to other models, the sensitivity to both high and low risk candidates is beneficial and even if some additional candidates 
who were low risk were flagged as high risk, at least manual review could then determine the final outcome versus a model that has less sensitivity to high risk
candidates and would recommend loans to those who may actually be more likely to default.


