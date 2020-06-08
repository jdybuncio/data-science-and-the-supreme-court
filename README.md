## Using Oral Arguments to Predict Supreme Court Case Outcomes

The United States Supreme Court has also had to adjust due to the global pandemic caused by COVID-19. While the court is still in session, oral arguments are being covered live for the first time and are taking place virtually via teleconference [source](https://www.nytimes.com/2020/05/03/us/politics/supreme-court-coronavirus.html). And while past sessions weren't covered live, it turns out that all oral arguments dating back to the 1950's have been transcribed and made publicly available via pdf format. 

This project seeks to use data from oral arguments to create models which can predict whether the Petitioner will win a given Supreme Court Case.

*by JDyBuncio*
*6/8/2020*

## Directory Structure & Replication

The Supreme Court data I used is stored as two types of JSON files: one containing the transcript from oral arguments and the other containing details of the case, such as its outcome. 10 examples of each of these types of JSON files are contained in this repository's ```./data/cases``` directory. The entire library of oral arguments I used can be obtained and copied over from the following repository: https://github.com/walkerdb/supreme_court_transcripts.git in its ```./oyez/cases``` directory. 

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
  - [Background](#background)
  - [The Data](#the-data)
  - [Question and Hypothesis](#question-and-hypothesis)
  - [Methodology](#methodology)
- [Exploratory Data Analysis](#exploratory-data-analysis-highlights)
- [Model Selection](#model-selection)
  - [Test Metric](#test-metric)
  - [Model Evaluation](#model-evaluation)
  - [Hyperparameter Tuning](#feature-importance)
  - [Results and Interpretation](#results-and-interpretation)
- [Conclusion](#conclusion)

# Introduction

## Background



## Other

<p align="center">
  <img src="images/timeline.png" width = 400>>
</p>

<p align="center">
  <img src="images/parties.png" width = 400>>
</p>


<p align="center">
  <img src="images/histogram_cases.png" width = 400>>
</p>

<p align="center">
  <img src="images/dataframe_workflow.png" width = 400>>
</p>

<p align="center">
  <img src="images/class_balance.png" width = 400>>
</p>

<p align="center">
  <img src="images/EDA_Diff_Questions.png" width = 400>>
</p>

<p align="center">
  <img src="images/frequent_words.png" width = 400>>
</p>


<p align="center">
  <img src="images/grid_search.png" width = 400>>
</p>


<p align="center">
  <img src="images/precision_recall_curve.png" width = 400>>
</p>


<p align="center">
  <img src="images/final_results_table.png" width = 400>>
</p>

<p align="center">
  <img src="images/final_results.png" width = 400>>
</p>

<p align="center">
  <img src="images/feature_importance.png" width = 400>>
</p>


<p align="center">
  <img src="images/word_frequency.png" width = 400>>
</p>















