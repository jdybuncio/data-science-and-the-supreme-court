import pandas as pd
import numpy as np
import json
import os
import fnmatch
import string
from nltk.corpus import stopwords


def transcript_parser(file_name):
    """
    Summary:
        This is a function which parses an oral argument transcript files from my datasource, https://github.com/walkerdb/supreme_court_transcripts. 
        The transcript files are contained in the following repository in the github mentioned: ./oyez/cases/*t01.json

    Args:
        file_name (.json): Filepath name for an individual case's oral argument file

    Returns:
        ./transcript_csvs/*.csv: This function will create a dataframe for the oral argument and save it as a csv file in a ./trancript_csvs local repository 
    """

    path_to_check = file_name.replace('.-t01.json','.csv')
    if os.path.exists('./transcript_csvs/' + path_to_check):
        return None #no need to run if file already exists locally

    #open file and store in contents variable
    with open('cases/'+file_name) as f:
        contents = json.load(f)
        
    #1 - focus on data contained in sections
    try:
        sections = contents['transcript']['sections']
    except (TypeError, ValueError):
        return None #if there isn't a section in the transcript file, then can't do anything

    #2 - want to store output as a list of dicts
    output = []

    #3 - iterate through each section
    for i in range(len(sections)):

        #4 - iterate through each turn of speaking within each section - get speaker info
        try:
            turns = sections[i]['turns']
        except (TypeError, ValueError):
            return None #if there are no turns, then can't do anything
        
        for turn in range(len(turns)):
            try:
                speaker_dict = turns[turn]['speaker']
            except (TypeError, ValueError):
                speaker_dict = None
                
            try:
                speaker_name = speaker_dict['identifier']
            except (TypeError, ValueError):
                speaker_name = None
                
            try: 
                speaker_role = speaker_dict['roles'][0]['type']
            except (TypeError, ValueError, IndexError):
                speaker_role = None

            #5 - iterate through blocks of text speaker may have - get text info
            try:
                blocks = turns[turn]['text_blocks'] 
            except (TypeError, ValueError):
                return None #if there are no text blocks, can't do anything
            
            for block in range(len(blocks)):
                d = {}
                start = blocks[block]['start']
                stop = blocks[block]['stop']
                text = blocks[block]['text']


                d['file_name'] = file_name.replace('-t01.json','')
                d['section'] = i
                d['speaker_name'] = speaker_name
                d['speaker_role'] = speaker_role
                d['start'] = start
                d['stop'] = stop
                d['text'] = text

                output.append(d)
                
    df = pd.DataFrame(output, columns = ['file_name','section','speaker_name', 'speaker_role', 'start','stop','text'])
    df['total_time'] = np.where(df.stop - df.start>=0.0, df.stop - df.start, 0)
    df.to_csv('transcript_csvs/{}.csv'.format(file_name.replace('-t01.json','')), index=False)  
    

def section_label_generator(file_name, df_advocates):
    """
    Summary:
        Reads in a parsed transcript file and the df_advocates DataFrame to assign a label as to which party is represented at each section of the transcript

    Args:
        file_name (['string']): file name of parsed transcript file
        df_advocates ([dataframe]): df_advocates dataframe

    Returns:
        df: [dataframe]: Dataframe where each row is a section with the label of which party is represented
    """
    # 1 - Read in File
    transcript_file = pd.read_csv('transcript_csvs/' +file_name)
    
    # 2 - Ignore Judges lines
    transcript_file = transcript_file[transcript_file.speaker_role != 'scotus_justice']
    transcript_file.file_name = transcript_file.file_name.astype('str')
    
    # 3 - Merge with Advocates Table
    df = pd.merge(transcript_file, df_advocates, how = 'left', left_on = ['file_name','speaker_name'], right_on = ['file_name','advocate'])
    df = df[['file_name','section','advocate_label']]
    
    # Drop dupe rows to try to get one row w/label per section 
    df = df.drop_duplicates(subset= ['file_name','section','advocate_label'])
    
    # Drop if advocate label is NaN
    df = df.dropna(subset = ['advocate_label'])
    
    return df

def create_section_map_df(df_advocates):
    """
    Summary:
        Loop through all parsed transcript files to create one file where each file_number (Supreme Court case) where each section has a label of which side is speaking (if I have the speaker's role)
    
    Args:
        df_advocates ([dataframe]): df_advocates dataframe

    Returns:
        df: [dataframe]: Dataframe where each row is a section with the label of which party is represented for all Cases
        ./data/df_advocates_section_map.csv [csv]:  Saves Dataframe as csv
    """

    for idx,filename in enumerate(os.listdir('./transcript_csvs')):
        if idx == 0:
            df = section_label_generator(filename, df_advocates)
        else:
            try:
                df2 = section_label_generator(filename, df_advocates)
                df = pd.concat([df,df2], sort = False)
            except pd.errors.ParserError:
                continue
    
    df = df.reset_index(drop = True)
    df.file_name = df.file_name.astype('str')

    df.to_csv('./data/df_advocates_section_map.csv')
    print('Successfully created df_advocates_section_map in data directory')
    return df 


def add_labels_to_df(df, df_advocates_section_map):
    """
    Summary:
        Takes in a Oral Argument Transcript dataframe and also the Advocates Section Map dataframe to create a 'turn_label' column so that each sections has a label for which party is speaking

    Args:
        df ([dataframe]): Oral Argument Transcript Dataframe
        df_advocates_section_map ([dataframe]): Dataframe where each row is a section with the label of which party is represented for all Cases

    Returns:
        df_join [dataframe]: Dataframe which contains the Oral Argument Transcript for a case, with a label of which party is represented
    """


    #need to make sure file_name column is a str
    df.file_name = df.file_name.astype('str')
    
    df_join = pd.merge(df, df_advocates_section_map, how = 'left', left_on = ['file_name','section'], right_on = ['file_name','section'])
    
    df_join['turn_label'] = np.where((df_join.advocate_label.isna() == True) & (df_join.section%2 == 0) , 'behalf_petitioner', 
                                 (np.where((df_join.advocate_label.isna() == True) & (df_join.section%2 != 0) , 'behalf_respondent',df_join.advocate_label)))
    
    df_join = df_join.drop(['advocate_label'],axis = 1)   
    
    return df_join

def create_dfs_per_speaking_party(df):
    """
    Summary:
        Takes in a Oral Argument transcript dataframe and splits into several dataframes - one for each speaking party. I.e. a dataframe which just contains when the Petitioner is talking,
        when the Judge is talking to the petitioner, and so on

    Args:
        df ([dataframe]): Takes the Oral Argument transcript dataframe which has been put through the add_labels_to_df function

    Returns:
        Dataframes for each speaking party
    """
    mask_petitioner = (df.turn_label == 'behalf_petitioner')  & (df.speaker_role.isna() == True)
    mask_petitioner_justice = (df.turn_label == 'behalf_petitioner')  & (df.speaker_role.isna() == False)

    mask_respondent = (df.turn_label == 'behalf_respondent') & (df.speaker_role.isna() == True)
    mask_respondent_justice = (df.turn_label == 'behalf_respondent') & (df.speaker_role.isna() == False)

    mask_ac = (df.turn_label == 'behalf_ac_neutral')  & (df.speaker_role.isna() == True)
    mask_ac_justice = (df.turn_label == 'behalf_ac_neutral')  & (df.speaker_role.isna() == False)

    mask_ac_petitioner = (df.turn_label == 'behalf_ac_petitioner')  & (df.speaker_role.isna() == True)
    mask_ac_petitioner_justice = (df.turn_label == 'behalf_ac_petitioner')  & (df.speaker_role.isna() == False)

    mask_ac_respondent = (df.turn_label == 'behalf_ac_respondent')  & (df.speaker_role.isna() == True)
    mask_ac_respondent_justice = (df.turn_label == 'behalf_ac_respondent')  & (df.speaker_role.isna() == False)

    df_p = df[mask_petitioner]
    df_p_j = df[mask_petitioner_justice]
    
    df_r = df[mask_respondent]
    df_r_j = df[mask_respondent_justice]

    df_ac = df[mask_ac]
    df_ac_j = df[mask_ac_justice]

    df_ac_p = df[mask_ac_petitioner]
    df_ac_p_j = df[mask_ac_petitioner_justice]

    df_ac_r = df[mask_ac_respondent]
    df_ac_r_j = df[mask_ac_respondent_justice]
    return df_p, df_p_j, df_r, df_r_j, df_ac, df_ac_j, df_ac_p, df_ac_p_j, df_ac_r, df_ac_r_j



def get_features(df, want_corpus = True):
    """
    Summary:
        Given the dataframes from the create_dfs_per_speaking_party function, this function gets the features desired from the transcripts, 
        such as: talk time, words (excl punctuation), interruptions, and questions

    Args:
        df ([dataframe]): Dataframes for each speaking party
        corpus (bool, optional): Boolean in case you want to return the corpus, excluding punctuation . Defaults to False.

    Returns:
        talk_time [float]: total number of talk time (in seconds) from the party in a given case
        num_words [int]: total number of words
        num_questions [int]: total number of questions asked
        num_interruptions [int]: total number of times interrupted
        corpus [string]: corpus of words, excluding punctuation and stopwords
    """
    
    punctuation_ = set(string.punctuation)
    STOPWORDS = set(stopwords.words('english'))

    if len(df) == 0:
        if want_corpus == False:
            return 0.0, 0, 0, 0
        else:
            return 0.0, 0, 0, 0, ''
            
    else:
        talk_time  = round(df.sum()['total_time'],0)
    
        lst_of_phrases = list(df.text)
        num_interruptions = len([1 for phrase in lst_of_phrases if phrase[:-3:-1] == '--'])
        num_questions = len([1 for phrase in lst_of_phrases if phrase[:-2:-1] == '?'])

        corpus = ' '.join(lst_of_phrases)

        #remove punctuation
        for i in punctuation_:
            corpus = corpus.replace(i,'')
        #remove Stopwords
        for word in STOPWORDS:
            token = ' ' + word + ' '
            corpus = corpus.replace(token, ' ')
            corpus = corpus.replace(' ', ' ')
        #lower case
        corpus = corpus.lower()

        lst_of_words = corpus.split(' ')
        num_words = len(lst_of_words)

        if want_corpus == True:
            return talk_time, num_words, num_questions, num_interruptions, corpus
        else:
            return talk_time, num_words, num_questions, num_interruptions


def generate_features_for_model_df(file_name, df_advocates_section_map):
    """
    Summary:
       Takes in an Oral Transcript file and the df_advocates_section_map dataframe and yields the features created of the passed in Supreme Court Case in a dictionary

    Args:
        file_name ([str]): file_name (*.csv) of the oral argument transcript stored locally
        df_advocates_section_map ([dataframe]): Dataframe where each row is a section with the label of which party is represented for all Cases

    Returns:
        d [dict]: Dictionary of created features for the passed in case
        
    """
    df = pd.read_csv('transcript_csvs/' + file_name)
    

    df_join = add_labels_to_df(df, df_advocates_section_map)
    
    df_p, df_p_j, df_r, df_r_j, df_ac, df_ac_j, df_ac_p, df_ac_p_j, df_ac_r, df_ac_r_j = create_dfs_per_speaking_party(df_join)
    
    tt_p, w_p, q_p, i_p, c_p = get_features(df_p, want_corpus = True)
    tt_p_j, w_p_j, q_p_j, i_p_j, c_p_j = get_features(df_p_j, want_corpus = True)
    tt_r, w_r, q_r, i_r, c_r = get_features(df_r, want_corpus = True)
    tt_r_j, w_r_j, q_r_j, i_r_j, c_r_j = get_features(df_r_j, want_corpus = True)
    tt_ac, w_ac, q_ac, i_ac, c_ac = get_features(df_ac, want_corpus = True)
    tt_ac_j, w_ac_j, q_ac_j, i_ac_j, c_ac_j = get_features(df_ac_j, want_corpus = True)
    tt_ac_p, w_ac_p, q_ac_p, i_ac_p, c_ac_p = get_features(df_ac_p, want_corpus = True)
    tt_ac_p_j, w_ac_p_j, q_ac_p_j, i_ac_p_j, c_ac_p_j= get_features(df_ac_p_j, want_corpus = True)
    tt_ac_r, w_ac_r, q_ac_r, i_ac_r, c_ac_r = get_features(df_ac_r, want_corpus = True)
    tt_ac_r_j, w_ac_r_j, q_ac_r_j, i_ac_r_j, c_ac_r_j = get_features(df_ac_r_j, want_corpus = True)
    
    d = {}
    
    d['file_name'] = str(max(df.file_name))

    # Petitioner - Judge
    d['talk_time_petitioner'] = tt_p
    d['words_petitioner'] = w_p
    d['interruptions_petitioner'] = i_p
    d['corpus_petitioner'] = c_p

    d['talk_time_petitioner_justice'] = tt_p_j
    d['words_petitioner_justice'] = w_p_j
    d['questions_petitioner_justice'] = q_p_j
    d['corpus_petitioner_justice'] = c_p_j

    # Respondent - Judge
    d['talk_time_respondent'] = tt_r
    d['words_respondent'] = w_r
    d['interruptions_respondent'] = i_r
    d['corpus_respondent'] = c_r

    d['talk_time_respondent_justice'] = tt_r_j
    d['words_respondent_justice'] = w_r_j
    d['questions_respondent_justice'] = q_r_j
    d['corpus_respondent_justice'] = c_r_j

    # Amicus Neutral - Judge
    d['talk_time_amicus_neutral'] = tt_ac
    d['words_amicus_neutral'] = w_ac
    d['interruptions_amicus_neutral'] = i_ac
    d['corpus_amicus_neutral'] = c_ac

    d['talk_time_amicus_neutral_justice'] = tt_ac_j
    d['words_amicus_neutral_justice'] = w_ac_j
    d['questions_amicus_neutral_justice'] = q_ac_j    
    d['corpus_amicus_neutral_justice'] = c_ac_j

    # Amicus Petitioner - Judge
    d['talk_time_amicus_petitioner'] = tt_ac_p
    d['words_amicus_petitioner'] = w_ac_p
    d['interruptions_amicus_petitioner'] = i_ac_p
    d['corpus_amicus_petitioner'] = c_ac_p

    d['talk_time_amicus_petitioner_justice'] = tt_ac_p_j
    d['words_amicus_petitioner_justice'] = w_ac_p_j
    d['questions_amicus_petitioner_justice'] = q_ac_p_j  
    d['corpus_amicus_petitioner_justice'] = c_ac_p_j

    # Amicus Respondent - Judge
    d['talk_time_amicus_respondent'] = tt_ac_r
    d['words_amicus_respondent'] = w_ac_r
    d['interruptions_amicus_respondent'] = i_ac_r
    d['corpus_amicus_respondent'] = c_ac_r

    d['talk_time_amicus_respondent_justice'] = tt_ac_r_j
    d['words_amicus_respondent_justice'] = w_ac_r_j
    d['questions_amicus_respondent_justice'] = q_ac_r_j
    d['corpus_amicus_respondent_justice'] = c_ac_r_j
    return d
