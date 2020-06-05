import pandas as pd
import numpy as np
import json
import os
import fnmatch



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


def case_detail_parser(file_name):
    """
    Summary:
        This is a function which parses the case detail files from my datasource, https://github.com/walkerdb/supreme_court_transcripts. 
        The transcript files are contained in the following repository in the github mentioned: ./oyez/cases/*.json (exclude files which end with t**.json)

    Args:
        file_name (.json): Filepath name for an individual case's non-oral argument file

    Returns:
        ./case_csvs/*.csv: This function will create a dataframe for the Case and save it as a csv file in a ./case_csvs local repository 
    """


    path_to_check = file_name.replace('.json','.csv')
    
    if os.path.exists('./case_csvs/' + path_to_check):
        return None #no need to run if file already exists locally

    with open('cases/'+file_name) as f:
        contents = json.load(f)
        
    d = {}
    d['file_name'] = file_name.replace('.json','')
    d['case_name']= contents['name']
    d['docket_number']= contents['docket_number']

    for idx,event in enumerate(contents['timeline']):
        key = event['event']
        d[key] = [event['dates'][0]] #just take the first date - there can be multiple per event
        
    d['first_party'] = contents['first_party']
    d['first_party_label'] = contents['first_party_label']
    d['second_party'] = contents['second_party']
    d['second_party_label'] = contents['second_party_label']
    
    if contents['decisions'] != None and contents['decisions'][0]['votes'] != None:
        d['winning_party'] =  contents['decisions'][0]['winning_party']
        d['decision_type'] = contents['decisions'][0]['decision_type']
        d['majority_vote'] = contents['decisions'][0]['majority_vote']
        d['minority_vote'] = contents['decisions'][0]['minority_vote']
        
        for justice in contents['decisions'][0]['votes']:
            key = justice['member']['identifier']
            d[key] = justice['vote']

    if contents['advocates'] != None:
        for idx,advocate in enumerate(contents['advocates']):
            if advocate['advocate'] != None:
                d['advocate_ID_' + str(idx)] = advocate['advocate']['ID']
                d['advocate_' + str(idx)] = advocate['advocate']['identifier']
                d['advocate_description_' + str(idx)] = advocate['advocate_description']
        
    
    df = pd.DataFrame(d)
    df.to_csv('case_csvs/{}.csv'.format(file_name.replace('.json','')), index=False)  
            

        

def number_cases():
    """Summary:
        One issue with the data source I'm using is that the two types of files: Case Details and Oral Arguments are in the same repository
        Both of these files are contained in the following repository in this github (https://github.com/walkerdb/supreme_court_transcripts)
        This function separates out the file names between the two types of files in order to generate a list of the Cases, using a: {year}.{docket #} format, 
        which have one oral argument only
    Returns:
        common_files [lst]: list of Supreme Court cases I will analyze in the format of: {year}.{docket #}. I am only looking at cases with one oral argument 
        since multiple oral arguments create an added complexity I am not working with for this project's case
    """

    #Set up empty lists to get the Case names in the {year}.{docket #}) format for all types of cases. Need to separate between Case files and Oral Argument Transcript files
    case_files = []
    case_transcript_files = []
    case_transcript_files2 = []
    case_transcript_files3 = []
    case_transcript_files4 = []
    case_USfiles = []

    for filename in os.listdir('./cases'):
        if filename.endswith("-t01.json"):
            case_transcript_files.append(filename)
        elif filename.endswith("-t02.json"):
            case_transcript_files2.append(filename)
        elif filename.endswith("-t03.json"):
            case_transcript_files3.append(filename) 
        elif filename.endswith("-t04.json"):
            case_transcript_files4.append(filename)  #no cases have more than 4 oral arguments in repo
        elif fnmatch.fnmatch(filename, '*us*'):
            case_USfiles.append(filename) #unclear what these files are. no documentation
        else:
            case_files.append(filename)

    #Hone in on cases with one oral argument and also one case file and get list of those
    lst = []
    for file in case_transcript_files:
        file = file.replace('-t01.json','')
        lst.append(file)
    lst2 = []
    for file in case_transcript_files2:
        file = file.replace('-t02.json','')
        lst2.append(file)
    lst3 = []
    for file in case_transcript_files3:
        file = file.replace('-t03.json','')
        lst3.append(file)
    lst4 = []
    for file in case_transcript_files4:
        file = file.replace('-t04.json','')
        lst4.append(file)    
        
    result = np.setdiff1d(np.array(lst), np.array(lst2), assume_unique=False)   
    result = np.setdiff1d(result, np.array(lst3), assume_unique=False)
    result = np.setdiff1d(result, np.array(lst4), assume_unique=False)
    
    case_lst = []
    for file in case_files:
        file = file.replace('.json','')
        case_lst.append(file)
        
    common_files = np.intersect1d(result, np.array(case_lst), assume_unique=False)
    
    print(f'There are {len(case_files)} total case files per this data pull. There are also {len(common_files)} cases with one oral argument which is set I will focus on')
    return common_files


def histogram_of_cases_by_yr(common_files):
    """
    Summary:
        Create histogram of number of cases w/ one oral argument by Docket Year

    Args:
        common_files ([list]): list of cases to be considered
    """
    years = []

    for file in common_files:
        years.append(file[0:4])

    fig, ax = plt.subplots(figsize = (20,10))
    ax.hist(years, bins = 65)
    plt.grid(True)
    plt.xticks(rotation=90)
    plt.title('Number of Cases w/ one oral argument in front of Supreme Court by Docket-Year')
    plt.xlabel('Docket-Year')
    plt.ylabel('Number of cases')
    plt.show()

    '''
    Note as to why num of cases go down over time:
        A 1988 law, the Supreme Court Case Selections Act, gave the court discretion over whether to hear appeals from circuit court decisions. 
        This gave the court more latitude over its caseload than it previously had, freeing it from hearing many cases that it previously was mandated to hear. 
        https://www.washingtonpost.com/news/monkey-cage/wp/2016/06/02/the-supreme-court-is-taking-far-fewer-cases-than-usual-heres-why/
    '''

def concat_case_details():
    """
    Summary:
        Using the created csvs containing the parsed Case Details, stored locally in the ./case_csvs repository, this function
        creates 1 data frame of all cases consolidated in one file
        It does this by:
            - putting all the Case Details into one file, where each row represents a Supreme Court Case
            

    Returns:
        df [dataframe]: Dataframe which contains a row for each case
        ./data/df_case_details_consolidated.csv [csv]:  Creates and saves a Dataframe which contains a row for each case 
    
    """

    # Read through each Case file from the ./case_csvs local repo. And put them all into one file so each case is one row
    for idx,filename in enumerate(os.listdir('./case_csvs')):
        
        if idx == 0:
            df = pd.read_csv('case_csvs/'+filename)
        else:
            try:
                df2 = pd.read_csv('case_csvs/'+filename)
                df = pd.concat([df,df2], sort = False)
            except pd.errors.ParserError:
                continue
    
    df.to_csv('./data/df_case_details_consolidated.csv')
    retrun df        
    


def create_df_with_label(df):
    """
    Summary:
        Using the df created from the concat_case_details function, this function creates a cleaned up Case dataframe with a label

    Returns:
        df_cases [data_frame]:  Dataframe which contains a row for each case and the outcome of if the petitioner won or not. Only contains cases which have a Decision
        ./data/df_case_details_labeled.csv [csv]:  Saves Dataframe as csv
    """

    df = df.drop_duplicates()
    cols = '''file_name
                case_name
                docket_number
                Argued
                Decided
                Granted
                first_party
                first_party_label
                second_party
                second_party_label
                winning_party
                decision_type
                majority_vote
                minority_vote
                advocate_0
                advocate_description_0
                advocate_1
                advocate_description_1
                advocate_2
                advocate_description_2
                advocate_3
                advocate_description_3
                advocate_4
                advocate_description_4
                advocate_5
                advocate_description_5
                advocate_6
                advocate_description_6
                advocate_7
                advocate_description_7
                advocate_8
                advocate_description_8
                advocate_9
                advocate_description_9
                advocate_10
                advocate_description_10
                advocate_11
                advocate_description_11
                advocate_12
                advocate_description_12
                advocate_13
                advocate_description_13
                advocate_14
                advocate_description_14
                advocate_15
                advocate_description_15'''
    justices = '''john_m_harlan2
                    hugo_l_black
                    william_o_douglas
                    potter_stewart
                    william_j_brennan_jr
                    byron_r_white
                    earl_warren
                    tom_c_clark
                    abe_fortas
                    thurgood_marshall
                    harry_a_blackmun
                    lewis_f_powell_jr
                    william_h_rehnquist
                    john_paul_stevens
                    sandra_day_oconnor
                    antonin_scalia
                    anthony_m_kennedy
                    david_h_souter
                    clarence_thomas
                    ruth_bader_ginsburg
                    stephen_g_breyer
                    felix_frankfurter
                    harold_burton
                    stanley_reed
                    sherman_minton
                    arthur_j_goldberg
                    sonia_sotomayor
                    john_g_roberts_jr
                    samuel_a_alito_jr
                    warren_e_burger
                    elena_kagan
                    neil_gorsuch
                    brett_m_kavanaugh
                    charles_e_whittaker'''
    cols = cols.replace(' ','')
    justices = justices.replace(' ','')
    df = df[cols.split('\n') + justices.split('\n')]
    df = df.sort_values(by = 'file_name')
    df = df.reset_index(drop = True)
        
    # remove cases which have no decision yet
    df = df[df['Decided'].isna() == False]
    
    # create label: petitioner_wins
    df['petitioner_wins'] = df.apply(lambda row: 1 if str(row['winning_party']) in str(row['first_party']) else 0, axis = 1)
    df.loc[(df.winning_party == 'Petitioner'),'petitioner_wins']=1

    #manually looked through 300 cases to create the label wherein it wasn't clear from the logic above who won or lost the case
    file_names_winning_party_change = ['2013.13-461',
            '1980.79-1429',
            '2010.10-238',
            '2008.08-146',
            '1986.85-693',
            '1987.86-279',
            '2013.12-138',
            '2018.17-1042',
            '1981.80-1002',
            '2008.07-1601',
            '2005.05-409',
            '2007.06-939',
            '1974.73-1256',
            '2014.13-719',
            '2006.06-278',
            '2007.06-666',
            '2005.05-130',
            '2014.14-86',
            '2013.12-1182',
            '2010.09-291',
            '2007.07-219',
            '2015.14-840',
            '2006.05-1589',
            '2016.15-1406',
            '1988.87-1716',
            '2013.11-681',
            '1974.73-1475',
            '1992.91-1600',
            '2002.01-950',
            '2007.06-1456',
            '1974.73-1210',
            '2005.04-1329',
            '2006.05-983',
            '2005.05-416',
            '2006.05-1448',
            '2005.04-1034',
            '2008.07-1015',
            '2005.04-1244',
            '2013.12-609',
            '2010.09-1205',
            '2007.06-1037',
            '2007.06-1195',
            '2013.12-3',
            '2003.02-1016',
            '2005.04-712',
            '2006.05-1284',
            '2018.17-1702',
            '2016.15-1248',
            '2013.12-1128',
            '2005.04-1371',
            '2009.09-475',
            '1982.82-354',
            '1976.75-1278',
            '2006.06-340',
            '1975.75-817',
            '2007.06-766',
            '2005.04-1618',
            '2018.17-1094',
            '1986.85-1722',
            '2014.13-271',
            '2010.09-6822',
            '2013.12-1315',
            '2010.09-993',
            '2007.06-1204',
            '2005.04-1332',
            '2006.05-1272',
            '2005.04-805',
            '2006.06-102',
            '1992.91-10',
            '2010.10-313',
            '2013.12-1036',
            '2006.05-1429',
            '2013.12-1038',
            '2013.12-1408',
            '2013.12-562',
            '1980.79-870',
            '2014.14-144',
            '2006.05-381',
            '2015.15-274',
            '1979.78-1078']

    for i in file_names_winning_party_change:
        df.loc[(df.file_name == i),'petitioner_wins']=1
    
    df.to_csv('./data/df_case_details_labeled.csv')  
    return df 
    


def create_case_dataframe(df):
    """
    Summary:
        Creates data frame from the create_df_with_label function for case information

    Returns:
        df_cases [data_frame]:  Dataframe which contains a row for each case and the outcome of if the petitioner won or not. Only contains cases which have a Decision
        ./data/df_cases.csv [csv]:  Saves Dataframe as csv
    """

    # Create df_cases dataframe
    df_cases = df[['file_name', 'case_name', 'docket_number', 'Argued', 'Decided',
       'first_party',  'second_party', 'majority_vote',
       'minority_vote','petitioner_wins']]
    
    df_cases.to_csv('./data/df_cases.csv')
    return df_cases


def create_judge_votes_dataframe(df):
    """
    Summary:
        Creates data frame from the create_df_with_label function for judge voting

    Returns:
        df_judge_votes [data_frame]: Dataframe which shows if each justice voted for the Petitioner or not in a given case
        ./data/df_judge_votes.csv [csv]:  Saves Dataframe as csv
        
    """
    justices = '''john_m_harlan2
                    hugo_l_black
                    william_o_douglas
                    potter_stewart
                    william_j_brennan_jr
                    byron_r_white
                    earl_warren
                    tom_c_clark
                    abe_fortas
                    thurgood_marshall
                    harry_a_blackmun
                    lewis_f_powell_jr
                    william_h_rehnquist
                    john_paul_stevens
                    sandra_day_oconnor
                    antonin_scalia
                    anthony_m_kennedy
                    david_h_souter
                    clarence_thomas
                    ruth_bader_ginsburg
                    stephen_g_breyer
                    felix_frankfurter
                    harold_burton
                    stanley_reed
                    sherman_minton
                    arthur_j_goldberg
                    sonia_sotomayor
                    john_g_roberts_jr
                    samuel_a_alito_jr
                    warren_e_burger
                    elena_kagan
                    neil_gorsuch
                    brett_m_kavanaugh
                    charles_e_whittaker'''
    justices = justices.replace(' ','')

    # Create df_judge votes dataframe
    df_judge_votes = df[['file_name', 'case_name','docket_number', 'petitioner_wins','john_m_harlan2',
       'hugo_l_black', 'william_o_douglas', 'potter_stewart',
       'william_j_brennan_jr', 'byron_r_white', 'earl_warren', 'tom_c_clark',
       'abe_fortas', 'thurgood_marshall', 'harry_a_blackmun',
       'lewis_f_powell_jr', 'william_h_rehnquist', 'john_paul_stevens',
       'sandra_day_oconnor', 'antonin_scalia', 'anthony_m_kennedy',
       'david_h_souter', 'clarence_thomas', 'ruth_bader_ginsburg',
       'stephen_g_breyer', 'felix_frankfurter', 'harold_burton',
       'stanley_reed', 'sherman_minton', 'arthur_j_goldberg',
       'sonia_sotomayor', 'john_g_roberts_jr', 'samuel_a_alito_jr',
       'warren_e_burger', 'elena_kagan', 'neil_gorsuch', 'brett_m_kavanaugh',
       'charles_e_whittaker']]
    
    #label when Judge votes for petitioner or not
    lst_judges = justices.split('\n')
    for justice in lst_judges:
        df_judge_votes.loc[(df_judge_votes[justice] == 'majority') & (df_judge_votes['petitioner_wins'] == 1),justice] = 1
        df_judge_votes.loc[(df_judge_votes[justice] == 'minority') & (df_judge_votes['petitioner_wins'] == 0),justice] = 1
        df_judge_votes.loc[(df_judge_votes[justice] == 'majority') & (df_judge_votes['petitioner_wins'] == 0),justice] = 0
        df_judge_votes.loc[(df_judge_votes[justice] == 'minority') & (df_judge_votes['petitioner_wins'] == 1),justice] = 0

    df_judge_votes.to_csv('./data/df_judge_votes.csv')
    return df_judge_votes


def create_advocate_dataframe(df):
    """
    Summary:
        Creates data frame from the create_df_with_label function for advocate information

    Returns:
        df_advocates [data_frame]: Dataframe which has details as to the Advocates of each case and whether they were arguing for the petitioner or respondent, or were a neutral party
        ./data/df_advocates.csv [csv]:  Saves Dataframe as csv
    """

    # Create df_advocates dataframe
    df_advocates = df[['file_name','advocate_0', 'advocate_description_0', 'advocate_1',
       'advocate_description_1', 'advocate_2', 'advocate_description_2',
       'advocate_3', 'advocate_description_3', 'advocate_4',
       'advocate_description_4', 'advocate_5', 'advocate_description_5',
       'advocate_6', 'advocate_description_6', 'advocate_7',
       'advocate_description_7', 'advocate_8', 'advocate_description_8',
       'advocate_9', 'advocate_description_9', 'advocate_10',
       'advocate_description_10', 'advocate_11', 'advocate_description_11',
       'advocate_12', 'advocate_description_12', 'advocate_13',
       'advocate_description_13', 'advocate_14', 'advocate_description_14',
       'advocate_15', 'advocate_description_15']]
    df_advocates =df_advocates.dropna(how='all')

    # stack all advocates into a 3 column dataframe
    for i in range(16):
        if i ==0:
            x1 = df_advocates[['file_name','advocate_'+str(i),'advocate_description_'+str(i)]]
            x1.columns = ['file_name','advocate', 'advocate_description']
        else:
            x2 = df_advocates[['file_name','advocate_'+str(i),'advocate_description_'+str(i)]]
            x2.columns = ['file_name','advocate', 'advocate_description']
            x1 = pd.concat([x1,x2],ignore_index = True)

    x1 =x1.dropna(subset=['advocate','advocate_description'])
    
    #Label when each advocate is arguing for the petitioner, respondent, or is a neutral party in a given case
        #appellant= petitioner
    x1['behalf_petitioner'] =  x1.apply(lambda x: 1 if ('petition' in str(x.advocate_description).lower()) or ('plaintiff' in str(x.advocate_description).lower())
                                    or ('ppella' in str(x.advocate_description).lower())
                                    else 0, axis=1)
        #appellee= respondent
    x1['behalf_respondent'] =  x1.apply(lambda x: 1 if ('respondent' in str(x.advocate_description).lower()) or ('defenda' in str(x.advocate_description).lower())
                                    or ('ppelle' in str(x.advocate_description).lower())    
                                    else 0, axis=1)
    x1['behalf_amicus_curiae'] =  x1.apply(lambda x: 1 if ('amicus curiae' in str(x.advocate_description).lower()) or ('amici curiae' in str(x.advocate_description).lower())
                                       else 0, axis=1)
    df_advocates = x1
    
    #Refine definition of which side advocate is on per case
        # Amicus Curiae on behalf of each side will have a 1 for both the side and the ac side. This creates 2 new cols.
    df_advocates['behalf_ac_petitioner'] = np.where(df_advocates.behalf_petitioner + df_advocates.behalf_amicus_curiae == 2, 1, 0)
    df_advocates['behalf_ac_respondent'] = np.where(df_advocates.behalf_respondent + df_advocates.behalf_amicus_curiae == 2, 1, 0)
        # Amicus Curiae neutral is one where only the AC col is checked. This creates 1 new col
    df_advocates['behalf_ac_neutral'] = np.where((df_advocates.behalf_respondent + df_advocates.behalf_petitioner+ df_advocates.behalf_amicus_curiae  == 1)
                                                & (df_advocates.behalf_amicus_curiae  == 1)
                                                , 1, 0)
        # Now you can isolate for non AC sides. Zero cols out when advocate is an AC. This edits 2 existing cols
    df_advocates['behalf_petitioner'] = np.where(df_advocates.behalf_petitioner + df_advocates.behalf_amicus_curiae == 2, 0, df_advocates.behalf_petitioner)
    df_advocates['behalf_respondent'] = np.where(df_advocates.behalf_respondent + df_advocates.behalf_amicus_curiae == 2, 0, df_advocates.behalf_respondent)

    #Only keep advocates wherein I know which side they argued for in a given case and simplify columns needed
    df_advocates['checker'] = df_advocates['behalf_petitioner'] + df_advocates['behalf_respondent'] +  df_advocates['behalf_ac_petitioner'] + df_advocates['behalf_ac_respondent'] + df_advocates['behalf_ac_neutral']
    df_advocates = df_advocates[df_advocates.checker == 1]
    df_advocates = df_advocates[['file_name','advocate','behalf_petitioner','behalf_respondent','behalf_ac_petitioner','behalf_ac_respondent','behalf_ac_neutral']]

    #Reverse dummy so I have One column which states which side Advocate argued for in a given case
    dummies = df_advocates[['behalf_petitioner','behalf_respondent','behalf_ac_petitioner','behalf_ac_respondent','behalf_ac_neutral']]
    s2 = pd.Series(dummies.columns[np.where(dummies!=0)[1]])
    advocate_labels = pd.DataFrame(s2, columns = ['advocate_label'])
    df_advocates = df_advocates.reset_index(drop = True)
    df_advocates = pd.merge(df_advocates, advocate_labels, how = 'inner',left_index = True, right_index = True)[['file_name','advocate','advocate_label']]

    df_advocates.to_csv('./data/df_advocates.csv')
    return df_advocates


if __name__ == "__main__":

    common_files = number_cases()
    #histogram_of_cases_by_yr(common_files)


    case_files_lst = []
    transcript_files_lst = []

    for file in common_files:
        case_file_name = file + '.json'
        transcript_file_name = file + '-t01.json'

        case_files_lst.append(case_file_name)
        transcript_files_lst.append(transcript_file_name)

    for file_name in case_files_lst:
        case_detail_parser(file_name)
    for file_name in transcript_files_lst:
        transcript_parser(file_name)

    df = concat_case_details()
    df = create_df_with_label(df)
    df_cases = create_case_dataframe(df)
    df_judge_votes = create_judge_votes_dataframe(df)
    df_advocates = create_advocate_dataframe(df)





''' Code to plot graphs of votes by justice
fig, axs = plt.subplots(6,6,figsize=(20,10))
for i, ax in enumerate(axs.flatten()):
    if i == 34 or i == 35:
        continue
    else:
        df.groupby(lst_of_justices[i]).size().plot.bar(x='decision', y='num_votes', title = lst_of_justices[i], ax = ax)
    
'''

# df['Argued'] = pd.to_datetime(df['Argued'], unit='s').dt.to_period('M')

