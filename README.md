## Using Oral Arguments to Predict Supreme Court Case Outcomes

The United States Supreme Court has also had to adjust due to the global pandemic caused by COVID-19. While the court is still in session, oral arguments are being covered live for the first time and are taking place virtually via teleconference [source](https://www.nytimes.com/2020/05/03/us/politics/supreme-court-coronavirus.html). And while past sessions weren't covered live, it turns out that all oral arguments dating back to the 1950's have been transcribed and made publicly available via pdf format. 

This project processes transcribed data from oral arguments to create models which can predict whether the Petitioner will win a given Supreme Court Case or not. While the contents of this repository convert the oral arguments into various numerical features to test various models, I find that no model relying on just data from oral arguments can beat a strategy of predicting the Petitioner wins everytime. This goes against research done - albeit on smaller datasets -  which claim that the side which receives more questions tends to lose at a higher rate [source](https://www.nytimes.com/2009/05/26/us/26bar.html?smid=nytcore-ios-share).

*by JDyBuncio*
*6/8/2020*


## Table of Contents
- [Introduction](#introduction)
  - [Directory Structure and Replication](#directory-structure-and-replication)
  - [Supreme Court Intro](#supreme-court-intro)
  - [The Data](#the-data)
  - [Hypothesis](#hypothesis)
- [Exploratory Data Analysis](#exploratory-data-analysis-highlights)
- [Model Selection](#model-selection)
  - [Test Metric](#test-metric)
  - [Model Evaluation](#model-evaluation)
  - [Hyperparameter Tuning](#feature-importance)
  - [Results and Interpretation](#results-and-interpretation)
- [Conclusion](#conclusion)

# Introduction

What the Supreme Court decides is binding law for all courts in the United States. They represent the highest rungs of the Judicial Branch and thus, scholars dedicate themselves to predicting how the court and its 9 Justices will react. Since a 1988 law which gave the Supreme Court added discretion over their caseload, the Supreme Courts hears arguments for 60-70 cases per year.

The contents of this repository process the transcriptions of oral arguments into numerical features to see if they add any signal to one's ability to predict the winning side of a Supreme Court case and test the hypothesis that the side which receives more questions is a predictor of which side will lose.


## Directory Structure and Replication

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

Required Packages: Python 3, pandas, numpy, json, os, sklearn, collections, seaborn, matplotlib, tensorflow

[Back to Top](#Table-of-Contents)

## Supreme Court Intro

Cases come to the Supreme Court in two ways: cases in which the court has Original Jurisdiction, which are those involving  Ambassadors, Public Ministers, and States, and cases in which the court has Ultimate Appellate Jurisdiction, which are those stemming from appeals to decisions made by the lower courts. The Court discusses cases during their weekly Conferences and take on cases when 4 out of the 9 judges agree to hear a case. 

* **Timeline of a Supreme Court Case**
<p align="center">
  <img src="images/timeline.png" width = 600>
</p>

* **Parties involved in a Supreme Court Case**
<p align="center">
  <img src="images/parties.png" width = 400>
</p>

Supreme Court cases consist of:
* A Petitioner - the side which brings the case to the court
* A Respondent - the side responding to the Petitioner
* The Justices - the 9 Supreme Court Justices who make up the Court

Once a case is Granted, there is usually around 160 days before it is argued in front of the Supreme Court. In this in-between time period, both sides of the case submit written briefings which have page limits. Oral Arguments are then heard with each side given a chance to make their argument and answer questions from the Justices. These are timed sessions which usually last for 1 hour. The most common structure is for the Petitioner to address the Justices and answer questions. The Respondent is then given an opportunity to do the same. And then the Petitioner usually is given a chance to rebuttal directly to the Justices. The Petitioner and Respondent do not address one another. After oral arguments, the case is discussed during the Court's Weekly Conferences and, though it varies, Decisions are given around 90 days after a case is argued.


[Back to Top](#Table-of-Contents)

## The Data

The dataset I use consists of over 6,000 Supreme Court Cases with transcribed oral arguments dating back to 1955. After removing cases which have missing data, and only looking at Cases which have one oral argument, I am left with a sample of 5,567 Supreme Court Cases. The distirbutuon of cases by Year is shown in the following histogram. The decrease in Cases per Year which starts in 1988 is due to the passing of the Supreme Court Case Selections Act which gave the court additional discretion of the cases they choose to take/pass.

<p align="center">
  <img src="images/histogram_cases.png" width = 500>
</p>

The following depicts the structure in how I parse the transcriptions of oral arguments into numerical data, such as the number of words, questions, interruptions, and total talk time each party has. I used the Case level data to find labels for both the speakers - so I can identify when the Petitioner, Respondent, or Justice is speaking, and who they are speaking to - also for which side wins the case.

<p align="center">
  <img src="images/dataframe_workflow.png" width = 600>
</p>


[Back to Top](#Table-of-Contents)

## Hypothesis

I hypothesize that, using data just from oral arguments, I will be able to create a prediction model for if the Petitioner Wins a case which can beat a Petitioner always wins strategy. There have been some studies, based on smaller datasets, which show that the side which receives more questions tends to lose more which makes me hopeful I can find some signal [source](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1373965)

[Back to Top](#Table-of-Contents)

# Exploratory Data Analysis Highlights

The Petitioner wins in 63% of the Cases I have in my sample, and the Respondent wins in 37% of cases. This follows in-line with what is known. This is logical given that a Case, which the Petitioner brings to the Court, requires 4 votes by the Justices to be Granted and then only 5 votes to have the majority and win the case - though a lot can happen from the time a case is granted and there are several examples of a Petitioner getting no votes when the Decision comes. 

Due to this imbalance, I make sure to stratify my data when I apply a Train-Test split to maintain this same class balance in my Train and Test sets.

* **Class Balance**

<p align="center">
  <img src="images/class_balance.png" width = 400>
</p>


The following two graphs take the difference in Questions and Interruptions by the Justices to the Petitioner and to the Respondent and assigns each case to one of five buckets. The graphs show the Petitioner Win Rate % in each of these buckets. For example, Petitioners who received between 14 and 127 more questions than the Respondent won 51% of cases, which is relative to a win rate of 73% for Petitioners who received 13 or less questions that the Respondent.

* **Relationships to Petitioner Winning Rates across some demographic variables**

<p align="center">
  <img src="images/EDA_Diff_Questions.png" width = 400>
</p>

While I convert transcripts to numerical data, I also conserved the transcriptions to be able to apply NLP and LSTM techniques. The following shows the 20 most frequent words used in oral arguments after removing Stop Words. The <OOV> is a catch call for words outside of the 10,000 most-used ones.

* **20 most frequent words found**

<p align="center">
  <img src="images/frequent_words.png" height = 200>
</p>

[Back to Top](#Table-of-Contents)

# Model Selection

## Test Metric

The metric I chose to evaluate my models was to optimize the **F1-Score** since I care more about the positive class and want to:
* Maximize Recall: Predict maximum % of Cases where the Petitioner Wins
* Maxmize Precision: Maximize the % predicted Petitioner Wins actually are cases in which the Petitioner Wins

[Back to Top](#Table-of-Contents)

## Model Evaluation
I used Cross Validation to evaluate the F1 of each of my models in order to direct my hyperparameter tuning and feature selection. 

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

