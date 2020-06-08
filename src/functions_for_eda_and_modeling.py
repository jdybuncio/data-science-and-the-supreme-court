import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn import tree
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def plot_bar(column_to_compare, df, ax):
    """
    Summary:
        Takes a feature and splits it into 5 bins and finds Avg Petitioner Win Rate for each bin

    Args:
        column_to_compare ([dataframe column]): Column you want to split into Bins and find Petitioner Win Rate for
        df ([dataframe]): modeling dataframe 
        ax ([pyplot ax]): axis we will plot on
    """

    if 'talk_time' in column_to_compare:
        x = pd.qcut(round(df[column_to_compare]/60,0),q=[0, .2, .4, .6, .8, 1],)
    else:
        x = pd.qcut(round(df[column_to_compare],0),q=[0, .2, .4, .6, .8, 1],)
        
    y = df.petitioner_wins
    
    cross_tab = pd.concat([x,y], axis=1).groupby(column_to_compare).mean()

    g = sns.barplot(x=cross_tab.index, y="petitioner_wins", data=cross_tab, ax = ax)
    ax.set_ylabel('% Petitioner Wins')
    ax.set_title(f'Petitioner Winning % by {column_to_compare} buckets')
    
    ax=g

    #annotate axis = seaborn axis
    for p in ax.patches:
                 ax.annotate("%.2f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', fontsize=11, color='gray', xytext=(0, 20),
                     textcoords='offset points')
    _ = g.set_ylim(0,1) #

def find_words_per_argument(df_model):
    """
    Summary:
        For each type of Corpus, find the number of Words, Avg Words per Case, and Avg Unique Words per Case

    Args:
        df_model ([dataframe]): Modeling dataframe

    Returns:
        d [dict]: dictionary showing the Word count totals corresponding to each type of corpus
    """

    cols = list(df_model.columns)

    # Get all Corpus Columns
    corpus_columns = []
    for col in cols:
        if col[0:3] == 'cor':
            corpus_columns.append(col)

    # Get total words and unique words per type of Corpus        
    d = {}
    for col in corpus_columns:
        df_model[col] = df_model[col].astype('str')


        results = set()
        df_model[col].str.split().apply(results.update)
        total_unique_words = len(results)

        lst_of_corpi = list(df_model[col])
        
        num_words_per_case =[]
        num_unique_words_per_case = []
        
        for one_case in lst_of_corpi:
            try:
                lst_of_words = one_case.split(' ')
            except (AttributeError):
                lst_of_words = ''
                
            num_words_per_case.append(len(lst_of_words))
            num_unique_words_per_case.append(len(Counter(lst_of_words).keys()))
            
        avg_words_per_case = np.mean(num_words_per_case)
        unique_words_per_case = np.mean(num_unique_words_per_case)
        
        d[col] = (total_unique_words, round(avg_words_per_case,0),round(unique_words_per_case,0))
    return d

def train_validation_split(df_model, cols_to_include, label = 'petitioner_wins', test_size_param = 0.20, shuffe_param = True):
    """
    Summary:
        Splits modeling dataframe into a Train and Test set dataframes

    Args:
        df_model ([dataframe]): Modeling dataframe
        cols_to_include ([list]): Columns you want to include in modeling
        label (str, optional): Defined Label.  Defaults to 'petitioner_wins'.
        test_size_param (float, optional): test set size.  Defaults to 0.20.
        shuffle_param (bool, optional): whether to shuffle data prior to splitting.  Defaults to True.

    Returns:
        X_train, X_test, y_train, y_test ([dataframes])
    """
    
    df = df_model[cols_to_include]
    y = df.pop(label)
    X = df
    X = X.fillna(0)

    #Stratify to maintain same balance and also shuffle before splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size_param, shuffle = shuffe_param, stratify = y, random_state = 42)
    print(f'Num cases in Train: {len(X_train)}. \n Train Class Balance: \n {y_train.value_counts() / len(y_train)}')
    print(f'Num cases in Test: {len(X_test)}. \n Test Class Balance: \n {y_train.value_counts() / len(y_train)}')

    X_train = X_train.reset_index(drop = True)
    X_test = X_test.reset_index(drop = True)
    y_train = y_train.reset_index(drop = True)
    y_test = y_test.reset_index(drop = True)

    return X_train, X_test, y_train, y_test

def create_tfid_features(X_train, X_test, cols_to_vectorize = 'corpus_petitioner', max_features = 10000):
    """
    Summary:
        Creates Tf-id feature set

    Args:
        X_train ([dataframe]): X_train dataframe with a text column
        X_test ([dataframe]): X_test dataframe with a text column
        cols_to_vectorize (str, optional): Text column to vectorize. Defaults to 'corpus_petitioner'.
        max_features (int, optional): Number of features (words) to vectorize text column into. Defaults to 10000.

    Returns:
        X_train_v, X_test_v [dataframes]: Feature dataframes with tf-id vectorized columns
    """
    # Create Tfid Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)

    #Isolate Feature to vectorize
    corpus_train = X_train[cols_to_vectorize]
    corpus_test = X_test[cols_to_vectorize]
    
    # Vectorize Feature
    train_corpus_vectorized = vectorizer.fit_transform(corpus_train)
    test_corpus_vectorized = vectorizer.transform(corpus_test)
    
    #Get Feature names
    features = vectorizer.get_feature_names()

    #Convert to Array
    train_tfid_arr = train_corpus_vectorized.toarray()
    test_tfid_arr = test_corpus_vectorized.toarray()

    df_X_train = pd.DataFrame(train_tfid_arr, columns = features)
    df_X_test = pd.DataFrame(test_tfid_arr, columns = features)
    
    X_train = X_train.drop([cols_to_vectorize],axis = 1)
    X_test = X_test.drop([cols_to_vectorize],axis = 1)
    

    X_train_v = pd.merge(X_train,df_X_train, left_index=True, right_index=True, how = 'left')
    X_test_v = pd.merge(X_test,df_X_test, left_index=True, right_index=True, how = 'left')

    X_train_v = X_train_v.fillna(0)
    X_test_v = X_test_v.fillna(0)

    return X_train_v, X_test_v

def scaler(X_train, X_test):
    """
    Summary:
        Centers and scales Features

    Args:
        X_train ([dataframe])
        X_test ([dataframe])

    Returns:
        scaler, X_train_scaled, X_test_scaled: scaler returned in case need to inverse_transform
    """
    columns = list(X_train.columns)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled = pd.DataFrame(X_train_scaled, columns = columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns = columns)

    return scaler, X_train_scaled, X_test_scaled

def model_selection(X_train, y_train, model, model_params, cv_param = 5, scoring_param = 'f1' ):
    """
    Summary:
        Takes a given list of models and model parameters to perform grid search cross validation to optimize a given score, given dataframes to fit on

    Args:
        X_train ([dataframe]): Features Dataframe to train on
        y_train ([dataframe]): Labels to train on
        model ([list]): List of Models to Test
        model_params ([set]): Set of parameters to Test
        cv_param (int, optional): Number of folds to perform cross validation on. Defaults to 5.
        scoring_param (str, optional): Score to optimize. Defaults to 'f1'.

    Returns:
        tuned_model[list]: list of best models from grid search
        f1_score[list]: list of corresponding f1_scores
    """
    # check if there are any non-numerical columns to drop
    categorical_cols = [f for f in X_train.columns if X_train.dtypes[f] == 'object']
    X_train = X_train.drop(categorical_cols, axis = 1)
    # X_test = X_test.drop(categorical_cols, axis = 1)

    # Perform Grid Search across params
    grdsearch_models = GridSearchCV(model, model_params, cv= cv_param, n_jobs = -1 ,  scoring = scoring_param, verbose = True)
    grdsearch_result = grdsearch_models.fit(X_train, y_train)

    #store this somewhere
    tuned_model = grdsearch_result.best_estimator_

    # F1 Score
    f1_score = grdsearch_result.best_score_

    return tuned_model, f1_score

def test_evaluation(X_train, X_test, y_train, y_test, fit_model):
    """
    Summary:
        Tests Model against Baseline Predictions across core evaluation metrics (F1, Recall, Precision)

    Args:
        X_train ([dataframe])
        X_test ([dataframe]) 
        y_train ([dataframe])
        y_test ([dataframe]) 
        fit_model ([fit classifier])

    Returns:
        df [dataframe]: Dataframe with results aggregated
    """

    #Baseline model - predict Petitioner always wins
    baseline_predictions = np.ones(len(y_test))
    
    f1_baseline = f1_score(y_test, baseline_predictions)
    recall_baseline = recall_score(y_test, baseline_predictions)
    precision_baseline = precision_score(y_test, baseline_predictions)

    baseline_predictions_train = np.ones(len(y_train))

    f1_baseline_train = f1_score(y_train, baseline_predictions_train)
    recall_baseline_train = recall_score(y_train, baseline_predictions_train)
    precision_baseline_train = precision_score(y_train, baseline_predictions_train)

    #Predictions of Test Data from fit model
    test_predictions = fit_model.predict(X_test)

    f1_model = f1_score(y_test, test_predictions)
    recall_model = recall_score(y_test, test_predictions)
    precision_model = precision_score(y_test, test_predictions)

    train_predictions = fit_model.predict(X_train)

    f1_model_train = f1_score(y_train, train_predictions)
    recall_model_train = recall_score(y_train, train_predictions)
    precision_model_train = precision_score(y_train, train_predictions)
    
    df = pd.DataFrame(np.array([[f1_baseline_train, recall_baseline_train, precision_baseline_train], 
                                [f1_model_train, recall_model_train, precision_model_train],
                                [f1_baseline, recall_baseline, precision_baseline], 
                                [f1_model, recall_model, precision_model]
                                ]),
               columns=['F1', 'Recall', 'Precision'], 
               index = ['Baseline_Train', 'Model_Train', 'Baseline_Test', 'Model_Test'])
    df = df.round(2)
    return df

def feature_importance(fit_classifier, X_train, X_test, y_train, y_test, color ='red'):
    """
    Summary:
        Get top 10 Feature Importance on fit model

    Args:
        fit_classifier ([fit classifier]): Fit Model
        X_train ([dataframe])
        X_test ([dataframe]) 
        y_train ([dataframe])
        y_test ([dataframe])
        color (str, optional): Defaults to 'red'.
    """
    
    importances = fit_classifier.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    cols = []
    values = []
    for f in range(10):

        idx = indices[f]
        col = X_train.columns[idx]
        value = importances[idx]
        cols.append(col)
        values.append(value)
   
    model_name = type(fit_classifier).__name__

    fig, ax = plt.subplots()
    y_pos = np.arange(len(cols))
    ax.barh(y_pos, values,
            color=color, align="center")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cols)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Feature Importance')
    ax.set_title('{0} - Top 10 features'.format(model_name))

    plt.show();

def plot_one_decision_tree(fit_classifier, X_train):

    # Extract single tree
    estimator = fit_classifier.estimators_[5]

    fig, ax = plt.subplots(figsize = (14,10))
    tree.plot_tree(estimator, feature_names = list(X_train.columns) , class_names = ['P_Win','P_Lose'],label = True, rounded = True, ax=ax, fontsize = 12)
    plt.show();

def calculate_threshold_values(y, prob):
    '''
    Build dataframe of the various confusion-matrix ratios by threshold
    from a list of predicted probabilities and actual y values
    '''
    df = pd.DataFrame({'prob': prob, 'y': y})
    df.sort_values('prob', inplace=True)
    
    actual_p = df.y.sum()
    actual_n = df.shape[0] - df.y.sum()

    df['tn'] = (df.y == 0).cumsum()
    df['fn'] = df.y.cumsum()
    df['fp'] = actual_n - df.tn
    df['tp'] = actual_p - df.fn

    df['fpr'] = df.fp/(df.fp + df.tn)
    df['tpr'] = df.tp/(df.tp + df.fn)
    df['precision'] = df.tp/(df.tp + df.fp)
    df = df.reset_index(drop=True)
    return df

def plot_precision_recall(ax, df):
    ax.plot(df.tpr,df.precision, label='precision/recall')
    #ax.plot([0,1],[0,1], 'k')
    ax.set_xlabel('recall')
    ax.set_ylabel('precision')
    ax.set_title('Precision/Recall Curve')
    ax.plot([0,1],[df.precision[0],df.precision[0]], 'k', label='random')
    ax.set_xlim(xmin=0,xmax=1)
    ax.set_ylim(ymin=0,ymax=1)
    
if __name__ == "__main__":
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

    # ,{'n_estimators': [100,200,500,1000],
    # 'learning_rate': [0.1, 0.05, .01],
    # 'max_depth': [3, 6, 10],  
    # 'min_child_weight': [3,6,10],  
    # 'objective': ['binary:logistic'],
    # 'gamma':[i/10.0 for i in range(0,5)],
    # 'eval_metric' : ['auc'],
    # 'nthread' : [4],
    # 'random_state': [42]}
    ]

    tuned_models = []
    f1_scores = []


    for model, model_params in zip(models_lst, model_params_lst):
        print(model)
        print(model_params)
        tuned_model, f1_score = model_selection(X_train_scaled, y_train, model, model_params, cv_param = 5, scoring_param = 'f1' )
        tuned_models.append(tuned_model)
        f1_scores.append(f1_score)
    
    index_of_max_f1 = np.argmax(f1_scores)
    best_model = tuned_models[index_of_max_f1]

    df  = test_evaluation(X_train_scaled, X_test_scaled, y_train, y_test, best_model)
    print(df.head())
    
    plot_one_decision_tree(best_model, X_train_scaled)
    feature_importance(best_model,  X_train_scaled, X_test_scaled, y_train, y_test)

    predicted_probas = best_model.predict_proba(X_test_scaled)
    df_confusion_matrix = calculate_threshold_values(y_test, probas[:,1])

    fig, ax = plt.subplots(figsize  = (14,8))
    plot_precision_recall(ax, df)
    plt.plot()



    ##### Modeling - NLP ##### 

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

    feature_importance(best_model_nlp,  X_train_v, X_test_v, y_train, y_test)
