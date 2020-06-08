
# import sys
# sys.path.append("..") # Adds higher directory to python modules path.

import pandas as pd
import numpy as np
import json
import os

from src.functions_to_create_case_dfs import case_detail_parser, number_cases, concat_case_details,create_df_with_label,create_case_dataframe,create_judge_votes_dataframe,create_advocate_dataframe
from src.functions_to_create_transcript_dfs import transcript_parser, section_label_generator, create_section_map_df, create_dfs_per_speaking_party, add_labels_to_df, get_features, generate_features_for_model_df
from src.functions_for_eda_and_modeling import train_validation_split, create_tfid_features, find_words_per_argument, scaler, model_selection, test_evaluation, feature_importance, plot_one_decision_tree, calculate_threshold_values, plot_precision_recall,plot_bar

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from collections import Counter
from sklearn import tree
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == "__main__":
    plt.style.use('ggplot')
    plt.ion()


    #########################################################
    ### Parsing Case Files into parts ### 
    #########################################################   
    
    # 1 - Get file names of cases you're looking into in a list
    common_files = number_cases()

    # 2 - Get list of Case Files and Transcript Files
    case_files_lst = []
    transcript_files_lst = []

    for file in common_files:
        case_file_name = file + '.json'
        transcript_file_name = file + '-t01.json'

        case_files_lst.append(case_file_name)
        transcript_files_lst.append(transcript_file_name)
    
    # 3 - Get Case Detail csvs created in local directory (need ./case_csvs directory created)
    for file_name in case_files_lst:
        case_detail_parser(file_name)
    print("Successfully parsed Case .json files")

    # 4 - Get consolidated info on cases (need ./data directory created)
    df = concat_case_details()
    print("Successfully Consolidated Case Files")

    # 5 -  Create labeled DF (need ./data directory created)
    df = create_df_with_label(df)

    # 6 -  Get case, judge_votes, and advocates DataFrames created (need ./data directory created)
    df_cases = create_case_dataframe(df)
    df_judge_votes = create_judge_votes_dataframe(df)
    df_advocates = create_advocate_dataframe(df)
    

    #########################################################
    ### Parsing Oral Argument Transcript Files into parts ###
    ########################################################## 

    # 1 - Get Transcript csvs created in local directory (need ./transcript_csvs directory created)
    for file_name in transcript_files_lst:
        transcript_parser(file_name)
    print("Successfully parsed Transcript .json files")

    # 2 - Create Section Map table so that each section has a label of which party is speaking
    df_advocates_section_map = create_section_map_df(df_advocates)

    # 3 - Get file names of parsed transcript files
    files_lst = []
    for filename in os.listdir('./data/transcript_csvs'):
        files_lst.append(filename)

    # 4 - Iterate through each transcript file to get features for each case
    lst = []
    for i, file in enumerate(files_lst):
        try:
            lst.append(generate_features_for_model_df(file, df_advocates_section_map))
        except (pd.errors.ParserError, TypeError):
            continue
    print("Successfully got features for each case")

    # 5 - Create modeling dataframe
    df_data = pd.DataFrame(lst)



    #########################################################
    ### Merge to create dataframe, with features, for modeling ###
    ########################################################## 
    df = pd.merge(df_cases, df_data, how = 'inner', on = 'file_name')

    #want to remove cases where petitioner and respondent both do not talk
    mask = (df.words_petitioner != 0) & (df.words_respondent != 0) & (df.words_petitioner_justice != 0) & (df.words_respondent_justice != 0) & (df.words_respondent_justice != 0)
    df = df[mask]

    #Create cols
    df['questions_diff'] = df.questions_petitioner_justice - df.questions_respondent_justice
    df['interruptions_diff'] = df.interruptions_petitioner - df.interruptions_respondent
    df['talk_time_lawyers_diff'] = df.talk_time_petitioner - df.talk_time_respondent
    df['talk_time_judges_diff'] = df.talk_time_petitioner_justice - df.talk_time_respondent_justice

    #remove 4 outliers
    df = df.drop(df[df['talk_time_petitioner']/60 > 300].index)
    df = df.drop(df[df['talk_time_petitioner_justice']/60 > 100].index)
    df = df.drop(df[df['talk_time_respondent_justice']/60 > 70].index)

    df.to_csv('./data/df_modeling.csv')
    print(f'Successfully created df_modeling in data directory. There are {len(df)} rows')
    
    #########################################################
    ### Run Numerical Models ### 
    #########################################################   
    
    df = pd.read_csv('data/df_modeling.csv', index_col = 0)


    cols = ['petitioner_wins', 'talk_time_petitioner', 
       'interruptions_petitioner', 
       'talk_time_petitioner_justice', 
       'questions_petitioner_justice', 
       'talk_time_respondent', 'interruptions_respondent',
       'talk_time_respondent_justice',
       'questions_respondent_justice',
       'questions_diff',
       'interruptions_diff', 'talk_time_lawyers_diff',
       'talk_time_judges_diff']
    
    # Graph variables listed above and split them into 5 bins and then find avg Petitioner Win Rate by each Bin
    fig, axs = plt.subplots(4,3, figsize = (20,15))
    for i, ax in enumerate(axs.flatten()):
        column_to_compare = cols[i+1]
        plot_bar(column_to_compare, df,ax)
    plt.tight_layout()
    plt.show()

    
    # Find total words, avg words per argument, and unique words per argument for each type of speaker direction
    dict_of_word_counts = find_words_per_argument(df)

    ##### Modeling - Numerical ##### 
    df_model = df.copy()
    cols_to_include = ['petitioner_wins', 'corpus_petitioner','talk_time_petitioner', 'words_petitioner',
        'interruptions_petitioner', 
        'talk_time_petitioner_justice', 'words_petitioner_justice',
        'questions_petitioner_justice', 
        'talk_time_respondent', 'words_respondent', 'interruptions_respondent',
        'talk_time_respondent_justice',
        'words_respondent_justice', 'questions_respondent_justice',
        'talk_time_amicus_neutral',
        'words_amicus_neutral', 'interruptions_amicus_neutral',
        'talk_time_amicus_neutral_justice',
        'words_amicus_neutral_justice', 'questions_amicus_neutral_justice',
        'talk_time_amicus_petitioner',
        'words_amicus_petitioner', 'interruptions_amicus_petitioner',
        'talk_time_amicus_petitioner_justice',
        'words_amicus_petitioner_justice',
        'questions_amicus_petitioner_justice',
        'talk_time_amicus_respondent',
        'words_amicus_respondent', 'interruptions_amicus_respondent',
        'talk_time_amicus_respondent_justice',
        'words_amicus_respondent_justice',
        'questions_amicus_respondent_justice',
        'questions_diff',
        'interruptions_diff', 'talk_time_lawyers_diff',
        'talk_time_judges_diff']

    # Train - Test Split on numerical cols
    X_train, X_test, y_train, y_test = train_validation_split(df_model, cols_to_include, label = 'petitioner_wins')

    # Train - Test Split for tf-id features
    X_train_v, X_test_v = create_tfid_features(X_train, X_test, cols_to_vectorize = 'corpus_petitioner', max_features = 5000)

    # Scale Features
    scaler_object, X_train_scaled, X_test_scaled = scaler(X_train, X_test)

    # Run Grid Search

    models_lst = [
            LogisticRegression(max_iter=10000), 
            RandomForestClassifier(), 
            GradientBoostingClassifier()
            # ,XGBClassifier()
             ]

    model_params_lst = [ 
    {'C': [0.1],
    'random_state': [42]},
    
    {'n_estimators': [500],
    'random_state': [42]},

   {'n_estimators': [500],
    'random_state': [42]}

    # More Extensive Grid Search - commented out for sake of default setting
    """
    model_params_lst = [ 
    {'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'solver' : ['newton-cg', 'lbfgs', 'liblinear'],
    'random_state': [42]},
    
    {'n_estimators': [100,200,500,1000],
    'max_features': ['sqrt', 0.25,.50,None],
    'min_samples_split': [5, 10, 50],        
    'min_samples_leaf': [5, 10, 50],
    'max_depth': [3, None],
    'bootstrap': [True, False],    
    'random_state': [42]},

   {'n_estimators': [100,200,500,1000],
    'learning_rate': [0.1, 0.05, .01],
    'max_features': ['sqrt', 0.25,.50,None],
    'min_samples_split': [5, 10, 50],        
    'min_samples_leaf': [5, 10, 50],
    'max_depth': [3, None],  
    'random_state': [42]}

#     # ,{'n_estimators': [100,200,500,1000],
#     # 'learning_rate': [0.1, 0.05, .01],
#     # 'max_depth': [3, 6, 10],  
#     # 'min_child_weight': [3,6,10],  
#     # 'objective': ['binary:logistic'],
#     # 'gamma':[i/10.0 for i in range(0,5)],
#     # 'eval_metric' : ['auc'],
#     # 'nthread' : [4],
#     # 'random_state': [42]}
    ]
    """
    

    tuned_models = []
    f1_scores = []


    for model, model_params in zip(models_lst, model_params_lst):
        print(model)
        print(model_params)
        tuned_model, f1_score = model_selection(X_train_scaled, y_train, model, model_params, cv_param = 3, scoring_param = 'f1' )
        tuned_models.append(tuned_model)
        f1_scores.append(f1_score)
    
    index_of_max_f1 = np.argmax(f1_scores)
    best_model = tuned_models[index_of_max_f1]

    df  = test_evaluation(X_train_scaled, X_test_scaled, y_train, y_test, best_model)
    print(df.head())

    model_name = type(best_model).__name__
    if model_name != 'LogisticRegression':
        plot_one_decision_tree(best_model, X_train_scaled)
        feature_importance(best_model,  X_train_scaled, X_test_scaled, y_train, y_test)

    predicted_probas = best_model.predict_proba(X_test_scaled)
    df_confusion_matrix = calculate_threshold_values(y_test, predicted_probas[:,1])

    fig, ax = plt.subplots(figsize  = (14,8))
    plot_precision_recall(ax, df_confusion_matrix)
    plt.plot()
    plt.show()
    
    #########################################################
    ### Run Numerical Models w/ NLP Features ### 
    #########################################################   
    # Commented Out for sake of default setting
    
    """ 

    models_lst = [
            LogisticRegression(max_iter=10000), 
            RandomForestClassifier(), GradientBoostingClassifier()]

    #Initial List
    model_params_lst = [ 
    {'C': [1],
    'solver' : ['lbfgs'],
    'random_state': [42],
    'n_jobs' : [-1]},
    
    {'n_estimators': [100], 
#     'max_features': ['sqrt', 0.25,.50,None],
#     'min_samples_split': [5, 10, 50],        
#     'min_samples_leaf': [5, 10, 50],
#     'max_depth': [3, None],
#     'bootstrap': [True, False],
     'n_jobs' : [-1],
    'random_state': [42]},

   {'n_estimators': [100],
#     'learning_rate': [0.1, 0.05, .01],
#     'max_features': [None],
#     'min_samples_split': [5, 10, 50],        
#     'min_samples_leaf': [5, 10, 50],
#     'max_depth': [3, None],  
    'random_state': [42]}
    ]

    tuned_models_nlp = []
    f1_scores_nlp = []


    for model, model_params in zip(models_lst, model_params_lst):
        print(model)
        print(model_params)
        tuned_model, f1_score = model_selection(X_train_v, y_train, model, model_params, cv_param = 3, scoring_param = 'f1' )
        tuned_models_nlp.append(tuned_model)
        f1_scores_nlp.append(f1_score)
    
    index_of_max_f1_nlp = np.argmax(f1_scores_nlp)
    best_model_nlp = tuned_models[index_of_max_f1_nlp]

    df_nlp  = test_evaluation(X_train_v, X_test_v, y_train, y_test, best_model_nlp)
    print(df_nlp.head())

    """
   

