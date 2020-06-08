## Using Oral Arguments to Predict Supreme Court Case Outcomes

The United States Supreme Court has also had to adjust due to the global pandemic caused by COVID-19. While the court is still in session, oral arguments are being covered live for the first time and are taking place virtually via teleconference [source](https://www.nytimes.com/2020/05/03/us/politics/supreme-court-coronavirus.html). And while past sessions weren't covered live, it turns out that all oral arguments dating back to the 1950's have been transcribed and made publicly available via pdf format. 

This project seeks to use this transcribed data from oral arguments to create models which can predict whether the Petitioner will win a given Supreme Court Case.

*by JDyBuncio*
*6/8/2020*

## Directory Structure & Replication

The Supreme Court data I used is stored as two types of JSON files: one containing the transcript from oral arguments and the other containing details of the case, such as its outcome. 10 examples of each of these types of JSON files are contained in this repository's ```./data/cases``` directory. The file name structure has the structure: ```{year}.{docket #}.json``` and oral argument transcripts append ```t01, t02, t03, and t04``` depending on the number of oral arguments tied to a case.

The entire library of oral arguments I used can be obtained from the following [repository](https://github.com/walkerdb/supreme_court_transcripts.git) in its ```./oyez/cases``` directory. 

The functions I used to parse, combine, and model this data are contained in this repository's ```src``` directory.

To replicate the parsing, dataframe creation, and model tuning that this repository covers, one can run the following:

```
#Clone this repository
git clone https://github.com/jdybuncio/data-science-and-the-supreme-court.git
cd data-science-and-the-supreme-court

#Run Script to create dataframe and to perform model tuning 
python create_df_and_fit_models_script.py
```

This will create the dataframe I used for modeling and identify the best prediction model given the data provided. An example of what the modeling dataframe looks like is in the data directory.

## Table of Contents
- [Introduction](#introduction)
  - [Supreme Court Intro](#supreme-court-intro)
  - [The Data](#the-data)
  - [Question and Hypothesis](#question-and-hypothesis)
- [Exploratory Data Analysis](#exploratory-data-analysis-highlights)
- [Model Selection](#model-selection)
  - [Test Metric](#test-metric)
  - [Model Evaluation](#model-evaluation)
  - [Hyperparameter Tuning](#feature-importance)
  - [Results and Interpretation](#results-and-interpretation)
- [Conclusion](#conclusion)

# Introduction

## Supreme Court Intro

<p align="center">
  <img src="images/timeline.png" width = 600>
</p>

<p align="center">
  <img src="images/parties.png" width = 400>
</p>

[Back to Top](#Table-of-Contents)

## The Data

<p align="center">
  <img src="images/histogram_cases.png" width = 400>>=
</p>

<p align="center">
  <img src="images/dataframe_workflow.png" width = 400>
</p>

[Back to Top](#Table-of-Contents)

## Question and Hypothesis




[Back to Top](#Table-of-Contents)
# Exploratory Data Analysis Highlights

* **Leakage Considered**

* **Class Balance**

<p align="center">
  <img src="images/class_balance.png" width = 400>
</p>

* **Relationships to Pass/Fail Rates across some demographic variables**

<p align="center">
  <img src="images/EDA_Diff_Questions.png" width = 400>
</p>


<p align="center">
  <img src="images/frequent_words.png" height = 200>
</p>

[Back to Top](#Table-of-Contents)
# Model Selection

## Test Metric

The metric I chose to evaluate my models was to optimize the **AUC** (area-under-the-curve) since I want to:
* Maximize the TPR: Predict maximum % students who Fail.
* Minimize the FPR: Minimize the % of students predicted to fail who Pass since doing so minimizes potential intervention costs.

[Back to Top](#Table-of-Contents)

## Model Evaluation
I used Cross Validation to evaluate the AUC of each of my models in order to direct my hyperparameter tuning and feature selection. 

I also used SKLearn's GridSearch to find the best values for the hyperparameters in my Random Forest and Boosting models.


<p align="center">
  <img src="images/grid_search.png" width = 400>
</p>

[Back to Top](#Table-of-Contents)

## Chosen Model
My Gradient Boost Model which included all the VLE interaction features I created had the highest AUC (0.77 - represented by the navy line in the image below). I got the same AUC when using my model to predict my validation set and also the testing set. The parameters of this model were:
n_estimators = 400, learning_rate = 0.2
                                      ,min_samples_split = 5
                                      ,min_samples_leaf = 100
                                      ,max_depth = 3
                                      ,max_features = 'sqrt'
                                      ,and subsample = 1.



<p align="center">
  <img src="images/final_results_table.png" width = 400>
</p>

<p align="center">
  <img src="images/final_results.png" width = 400>
</p>

<p align="center">
  <img src="images/precision_recall_curve.png" width = 400>
</p>


## Feature Importance
Below shows the Top 10 Features measured using SKLearn's feature importance from my chosen Gradient Boost model.

<p align="center">
  <img src="images/feature_importance.png" width = 400>
</p>

## Results and Interpretation
From my best Gradient Boost model, at a **threshold of 0.39**, I have:
* TPR (true-positive-rate), i.e. the percentage of students who fail that I accurately predict, to be **0.80**. 
* FPR (false-positive-rate), i.e. the percentage of students who pass that I predict to fail, to be **0.42**

So my model can predict 80% of the students who will Fail on the first day of class, but it will also  predict 42% of those who will pass, would Fail. The most important features in my model relate to the number of clicks a student has on the homepage before Day 1 of the class and the total number of days interacting with any of the VLE material before the first day of class.

[Back to Top](#Table-of-Contents)

# Conclusion
With a movement on-line, we have access to new student data which can help students, teachers and administrators better achieve success, however defined. This repository shows that tracked student behavior with class materials before a class officially begins is a strong predictor for student success and failure, relative to student demogrographics and class information for the given dataset observed. This is not surprising since interaction with materials before Day 1 could be a proxy which captures student preparedness and how serious one is taking the course, especially in an online environment where 30%+ of students eventually withdraw. Student interaction variables give us a way to capture this, unlike in in-person settings.

I speculate that as schools move increasingly online, more work will be done to mine the new data collected, such as that discussed in this repository, and will be used for new applications we otherwise were not privy to in in-person settings.

[Back to Top](#Table-of-Contents)

