import pandas as pd
import numpy as np
import json
import os

from src.functions_to_create_case_dfs import case_detail_parser, number_cases, concat_case_details,create_df_with_label,create_case_dataframe,create_judge_votes_dataframe,create_advocate_dataframe
from src.functions_to_create_transcript_dfs import transcript_parser, section_label_generator, create_section_map_df, create_dfs_per_speaking_party, add_labels_to_df, get_features, generate_features_for_model_df


if __name__ == "__main__":

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

    # 4 - Get consolidated info on cases (need ./data directory created)
    df = concat_case_details()

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

    # 2 - Create Section Map table so that each section has a label of which party is speaking
    df_advocates_section_map = create_section_map_df(df_advocates)

    # 3 - Get file names of parsed transcript files
    files_lst = []
    for filename in os.listdir('./transcript_csvs'):
        files_lst.append(filename)

    # 4 - Iterate through each transcript file to get features for each case
    lst = []
    for i, file in enumerate(files_lst):
        try:
            lst.append(generate_features_for_model_df(file, df_advocates_section_map))
        except (pd.errors.ParserError, TypeError):
            continue
    
    # 5 - Create modeling dataframe
    df_data = pd.DataFrame(lst)
    

    #########################################################
    ### Merge to create dataframe, with features, for modeling ###
    ########################################################## 
    df_modeling = pd.merge(df_cases, df_data, how = 'inner', on = 'file_name')
    df_modeling.to_csv('./data/df_modeling.csv')
    print(f'Successfully created df_modeling in data directory. There are {len(df_modeling)} rows')