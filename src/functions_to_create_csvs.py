import pandas as pd
import numpy as np
import json
import os



def transcript_parser(file_name):
        
    path_to_check = file_name.replace('.-t01.json','.csv')
    if os.path.exists('./transcript_csvs/' + path_to_check):
        return None #no need to run if file exists

    #open file and store in contents object
    with open('cases/'+file_name) as f:
        contents = json.load(f)
    
    if contents['transcript'] == None:
        return None
    if contents['transcript']['sections'] == None:
        return None
    #focus on the data contained in sections 
    sections = contents['transcript']['sections']
    
    #want to store output as a list of dicts
    output = []
    
    
    #iterate through each section
    for i in range(len(sections)):
        turns = sections[i]['turns']
        
        #iterate through each turn of speaking within each section
        for turn in range(len(turns)):
            d = {}
            d['file_name'] = file_name.replace('-t01.json','')
            d['section'] = i
            if turns[turn]['speaker'] == None:
                break
            else:
                speaker_dict = turns[turn]['speaker']

            if turns[turn]['text_blocks'][0] == None:
                break
            else:
                transcript_dict = turns[turn]['text_blocks'][0]

            if speaker_dict['identifier'] != None:
                speaker_name = speaker_dict['identifier']
            else:
                speaker_name = None

            if speaker_dict['ID'] != None:
                speaker_id = speaker_dict['ID']
            else:
                speaker_id = None


            
            if speaker_dict['roles'] == None:
                speaker_role = None
                
            if isinstance(speaker_dict['roles'], list):
                
                try:
                    speaker_role = speaker_dict['roles'][0]['type'] 
                except IndexError:
                    speaker_role = None                    
            else:
                speaker_role = None

            start = transcript_dict['start']
            stop = transcript_dict['stop']
            text = transcript_dict['text']
            d['speaker_id'] = speaker_id
            d['speaker_name'] = speaker_name
            d['speaker_role'] = speaker_role
            d['start'] = start
            d['stop'] = stop
            d['text'] = text
            output.append(d)
    df = pd.DataFrame(output, columns = ['file_name','section','speaker_id','speaker_name', 'speaker_role', 'start','stop','text'])
    df.to_csv('transcript_csvs_2/{}.csv'.format(file_name.replace('-t01.json','')), index=False)  
            
    

def case_detail_parser(file_name):
    
    
    path_to_check = file_name.replace('.json','.csv')
    if os.path.exists('./case_csvs/' + path_to_check):
        return None #no need to run

    
    with open('cases/'+file_name) as f:
        contents = json.load(f)
        
    d = {}
    d['file_name'] = file_name.replace('.json','')
    d['case_name']= contents['name']
    d['docket_number']= contents['docket_number']
#     d['addtl_docker_numbers'] = contents['additional_docket_numbers']

    for idx,event in enumerate(contents['timeline']):
        key = event['event']
        d[key] = [event['dates'][0]] #just take the first date - there can be multiple per event
        
    d['first_party'] = contents['first_party']
    d['first_party_label'] = contents['first_party_label']
    d['second_party'] = contents['second_party']
    d['second_party_label'] = contents['second_party_label']
    
    if contents['decisions'] != None and contents['decisions'][0]['votes']!=None:
        d['winning_party'] =  contents['decisions'][0]['winning_party']
        d['decision_type'] = contents['decisions'][0]['decision_type']
        d['majority_vote'] = contents['decisions'][0]['majority_vote']
        d['minority_vote'] = contents['decisions'][0]['minority_vote']
        
        for justice in contents['decisions'][0]['votes']:
            key = justice['member']['identifier']
            d[key] = justice['vote']
#     else:
#         d['winning_party'] =  None
#         d['decision_type'] = None
#         d['majority_vote'] = None
#         d['minority_vote'] = None

    if contents['advocates'] != None:
        for idx,advocate in enumerate(contents['advocates']):
            if advocate['advocate'] != None:
                d['advocate_ID_' + str(idx)] = advocate['advocate']['ID']
                d['advocate_' + str(idx)] = advocate['advocate']['identifier']
                d['advocate_description_' + str(idx)] = advocate['advocate_description']
        
    
    df = pd.DataFrame(d)
    df.to_csv('case_csvs/{}.csv'.format(file_name.replace('.json','')), index=False)  
            


def number_cases():
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
            case_transcript_files4.append(filename)      #no cases have more than 4 oral arguments
        elif fnmatch.fnmatch(filename, '*us*'):
            case_USfiles.append(filename)
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
    return case_files, common_files

def histogram_of_cases_by_yr(common_files):
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

    https://www.washingtonpost.com/news/monkey-cage/wp/2016/06/02/the-supreme-court-is-taking-far-fewer-cases-than-usual-heres-why/
    A 1988 law, the Supreme Court Case Selections Act, gave the court discretion over whether to hear appeals from circuit court decisions. This gave the court more latitude over its caseload than it previously had, freeing it from hearing many cases that it previously was mandated to hear. 


    '''

def concat_case_details():
    for idx,filename in enumerate(os.listdir('./case_csvs')):
        
        if idx == 0:
            df = pd.read_csv('case_csvs/'+filename)
        else:
            try:
                df2 = pd.read_csv('case_csvs/'+filename)
                df = pd.concat([df,df2], sort = False)
            except pd.errors.ParserError:
                continue
    
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

    df = df[cols.split('\n') + justices.split('\n')]
    df.to_csv('case_summary_agg_cleaned.csv')
    return df



if __name__ == "__main__":

    case_files, common_files = number_cases()
    histogram_of_cases_by_yr(common_files)


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