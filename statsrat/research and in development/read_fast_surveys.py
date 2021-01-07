import numpy as np
import pandas as pd
import glob


def promis_lookup(raw_sum, scale):
    """
    Lookup table to process promis measures (from promis scoring manuals).
    
    Arguments
    ---------
    raw_sum: list-like
        Raw sums of scores from promis short forms.
    scale: str
        Name of the scale ('anx', 'pos', 'ang' or 'dep').
    """
    anx_table = pd.Series([37.1, 43.2, 45.9, 47.8, 49.4, 50.8, 52.1, 53.2, 54.3, 55.4, 56.4, 57.4, 58.4, 59.4, 60.4, 61.4, 62.5, 63.5, 64.5, 65.6, 66.6, 67.7, 68.7, 69.8, 70.8, 71.9, 73.0, 74.1, 75.4, 76.7, 78.2, 80.0, 83.1],
                          index = np.arange(8, 40 + 1))
    pos_table = pd.Series([14.4, 15.7, 17.3, 18.7, 20.1, 21.3, 22.5, 23.5, 24.5, 25.4, 26.3, 27.1, 27.9, 28.7, 29.4, 30.2, 30.9, 31.6, 32.3, 33.0, 33.7, 34.4, 35.0, 35.7, 36.4, 37.0, 37.7, 38.3, 39.0, 39.6, 40.3, 40.9, 41.6, 42.2, 42.9, 43.5, 44.2, 44.8, 45.5, 46.2, 46.9, 47.5, 48.2, 48.9, 49.6, 50.3, 51.1, 51.8, 52.6, 53.4, 54.2, 55.0, 55.9, 56.9, 58.0, 59.2, 60.5, 62.1, 63.9, 66.3, 69.9],
                          index = np.arange(15, 75 + 1))
    ang_table = pd.Series([32.9, 38.1, 41.3, 44.0, 46.3, 48.4, 50.6, 52.7, 54.7, 56.8, 58.8, 60.8, 62.9, 65.0, 67.2, 69.4, 71.7, 74.1, 76.8, 79.6, 82.9],
                          index = np.arange(5, 25 + 1))
    dep_table = pd.Series([38.2, 44.7, 47.5, 49.4, 50.9, 52.1, 53.2, 54.1, 55.1, 55.9, 56.8, 57.7, 58.5, 59.4, 60.3, 61.2, 62.1, 63.0, 63.9, 64.9, 65.8, 66.8, 67.7, 68.7, 69.7, 70.7, 71.7, 72.8, 73.9, 75.0, 76.4, 78.2, 81.3],
                          index = np.arange(8, 40 + 1))
    table_dict = {'anx': anx_table, 'pos': pos_table, 'ang': ang_table, 'dep': dep_table}
    return table_dict[scale].loc[raw_sum].values

def read_surveys(path, fun, ident_col = None, header = 'infer'):
    """
    Read all surveys in 'path' and put into a single data frame.
    'fun' is the function used to process each individual's survey data.  It should output a dict.
    """
    # list .csv files in the directory
    file_set = [file for file in glob.glob(path + "**/*.csv", recursive=True)]
    assert len(file_set) > 0, 'Cannot find any files in specified path.'
    
    # loop through files
    n_f = len(file_set)
    results_list = []
    for i in range(n_f):
        try:
            # import raw data
            raw = pd.read_csv(file_set[i], error_bad_lines = False, header = header)
            results = fun(raw)
            if ident_col is None:
                    ident = 'sub_' + str(i)
            else:
                ident = raw[ident_col].dropna()[0]
                if not isinstance(ident, str): # change participant ID to string if it's not already a string
                    if ident.dtype == float:
                        ident = ident.astype(int)
                    ident = ident.astype(str)
            results['ident'] = ident
            results_list += [results]
        except Exception as e:
            print(e)
    
    # assemble results into a single data frame and return that
    survey_df = pd.DataFrame(results_list)
    survey_df = survey_df.set_index('ident', drop = True)
    return survey_df     

def process_psychopy(df):
    """
    Process survey data from a single Psychopy data file (that's already been import as a Pandas data frame).
    """
    names = 8*['anx_slider']
    for n in range(8):
        names[n] += str(n+1) + '.response'
    raw_promis_anx = df[names].sum().sum()
    promis_anx = promis_lookup(raw_promis_anx, 'anx')
    
    names = 5*['ang_slider']
    for n in range(5):
        names[n] += str(n+1) + '.response'
    raw_promis_ang = df[names].sum().sum()
    promis_ang = promis_lookup(raw_promis_ang, 'ang')
    
    names = 8*['dep_slider']
    for n in range(8):
        names[n] += str(n+1) + '.response'
    raw_promis_dep = df[names].sum().sum()
    promis_dep = promis_lookup(raw_promis_dep, 'dep')
    
    names = 15*['pos_slider']
    for n in range(15):
        names[n] += str(n+1) + '.response'
    raw_promis_pos = df[names].sum().sum()
    promis_pos = promis_lookup(raw_promis_pos, 'pos')
    
    names = 5*['bite_slider']
    for n in range(5):
        names[n] += str(n+1) + '.response'
    bite = df[names].sum().sum()
    
    # package together and return results
    # ADD REMAINING MEASURES LATER
    results = {'promis_anx': promis_anx,
               'promis_ang': promis_ang,
               'promis_dep': promis_dep,
               'promis_pos': promis_pos,
               'bite': bite}
               #'ders': ders,
               #'dass': dass,
               #'promis_sat': promis_sat,
               #'vocab': vocab}
    return results

def process_testable(df):
    """
    Process survey data from a single Testable data file (that's already been import as a Pandas data frame).
    """
    # PROMIS anxiety
    index = df.responseRows == 'I felt fearful; I found it hard to focus on anything other than my anxiety; My worries overwhelmed me; I felt uneasy; I felt nervous; I felt like I needed help for my anxiety; I felt anxious; I felt tense'
    responses = np.array(df['responseCode'].loc[index].values[0].split('_'), dtype = 'float')
    raw_promis_anx = responses.sum()
    promis_anx = promis_lookup(raw_promis_anx, 'anx')

    # PROMIS anger
    index = df.responseRows == 'I was irritated more than people knew; I felt angry; I felt like I was ready to explode; I was grouchy; I felt annoyed'
    responses = np.array(df['responseCode'].loc[index].values[0].split('_'), dtype = 'float')
    raw_promis_ang = responses.sum()
    promis_ang = promis_lookup(raw_promis_ang, 'ang')
    
    # PROMIS depression
    index = df.responseRows == 'I felt worthless; I felt helpless; I felt depressed; I felt hopeless; I felt like a failure; I felt unhappy; I felt that I had nothing to look forward to; I felt that nothing could cheer me up'
    responses = np.array(df['responseCode'].loc[index].values[0].split('_'), dtype = 'float')
    raw_promis_dep = responses.sum()
    promis_dep = promis_lookup(raw_promis_dep, 'dep')

    # PROMIS positive affect
    index = df.responseRows == 'I felt cheerful; I felt attentive; I felt delighted; I felt happy; I felt joyful; I felt enthusiastic; I felt determined; I felt interested; I was thinking creatively; I liked myself; I felt peaceful; I felt good-natured; I felt useful; I felt understood; I felt content'
    responses = np.array(df['responseCode'].loc[index].values[0].split('_'), dtype = 'float')
    raw_promis_pos = responses.sum()
    promis_pos = promis_lookup(raw_promis_pos, 'pos')

    # BITE
    index = df.responseRows == 'I have been grumpy; I have been feeling like I might snap; Other people have been getting on my nerves; Things have been bothering me more than they normally do; I have been feeling irritable'
    responses = np.array(df['responseCode'].loc[index].values[0].split('_'), dtype = 'float')
    bite = responses.sum()

    # DERS-18
    # FIX BULLSHIT CHARACTERS
    #index0 = df.responseRows == "I pay attention to how I feel.; I have no idea how I am feeling.; I have difficulty making sense out of my feelings.; I am attentive to my feelings.; I am confused about how I feel.; When I‚Äôm upset, I acknowledge my emotions."
    #responses0 = np.array(df['responseCode'].loc[index0].split(separator = '_'), dtype = 'float')
    #index1 = df.responseRows == "When I‚Äôm upset, I become embarrassed for feeling that way.; When I‚Äôm upset, I have difficulty getting work done.; When I‚Äôm upset, I become out of control.; When I'm upset, I believe that I will remain that way for a long time.; When I'm upset, I believe that I'll end up feeling very depressed.; When I'm upset, I have difficulty focusing on other things."
    #responses1 = np.array(df['responseCode'].loc[index1].split(separator = '_'), dtype = 'float')
    #index2 = df.responseRows == "When I'm upset, I feel ashamed with myself for feeling that way.; When I'm upset, I feel guilty for feeling that way.; When I'm upset, I have difficulty concentrating.; When I'm upset, I have difficulty controlling my behaviors.; When I'm upset, I believe that wallowing in it is all I can do.; When I'm upset, I lose control over my behaviors."
    #responses2 = np.array(df['responseCode'].loc[index2].split(separator = '_'), dtype = 'float')
    # FINISH.
    # reverse code the following items:
    # 1. I pay attention to how I feel
    # 4. I am attentive to my feelings
    # 6. When I’m upset, I acknowledge my emotions.
    
    # DASS-21
    #index0 = df.responseRows == "I found it hard to wind down;, I was aware of dryness of my mouth; I couldn't seem to experience any positive feeling at all; I experienced breathing difficulty (eg, excessively rapid breathing, breathlessness in the absence of physical exertion); I found it difficult to work up the initiative to do things; I tended to over-react to situations; I experienced trembling (eg, in the hands)"
    #responses0 = np.array(df['responseCode'].loc[index0].split(separator = '_'), dtype = 'float')
    #index1 = df.responseRows == "I felt that I was using a lot of nervous energy; I was worried about situations in which I might panic and make a fool of myself; I felt that I had nothing to look forward to; I found myself getting agitated; I found it difficult to relax; I felt down-hearted and blue; I was intolerant of anything that kept me from getting on with what I was doing"
    #responses1 = np.array(df['responseCode'].loc[index1].split(separator = '_'), dtype = 'float')
    #index2 = df.responseRows == "I felt I was close to panic; I was unable to become enthusiastic about anything; I felt I wasn't worth much as a person; I felt that I was rather touchy; I was aware of the action of my heart in the absence of physical exertion (eg, sense of heart rate increase, heart missing a beat); I felt scared without any good reason; I felt that life was meaningless"
    #responses2 = np.array(df['responseCode'].loc[index2].split(separator = '_'), dtype = 'float')
    # FINISH.

    # PROMIS satisfaction with social roles and activities
    # THERE'S A KEY ERROR, I.E. PRESUMABLY SOME DARN CHARACTER OR SOMETHING IS WRONG.
    #index = "I am satisfied with my ability to do things for my family.;, I am satisfied with my ability to do things for fun with others.;, I feel good about my ability to do things for my friends.;, I am satisfied with my ability to perform my daily routines.;, I am satisfied with my ability to do things for fun outside my home.;, I am satisfied with my ability to meet the needs of my friends.;, I am satisfied with my ability to do the work that is really important to me (include work at home).;, I am satisfied with my ability to meet the needs of my family."
    #responses = np.array(df['responseCode'].loc[index].values[0].split('_'), dtype = 'float')
    #promis_sat += responses.sum()

    # vocab test
    correct_answer = {'BEAST': 'animal',
                     'ADHESIVE': 'glue',
                     'DREARY': 'gloomy',
                     'MEAGER': 'bare',
                     'ADHERENCE': 'devotion',
                     'DEPRECATE': 'hate',
                     'HALCYON': 'peaceful',
                     'PACHYDERM': 'elephant',
                     'TRUTH': 'fact',
                     'REJOICE': 'revel',
                     'POTATION': 'drink',
                     'PIQUANT': 'nosy',
                     'ACOLYTE': 'church helper',
                     'COVENANT': 'pact',
                     'DORMANT': 'latent',
                     'CRUEL': 'mean',
                     'HERMETIC': 'completely sealed',
                     'CHUCKLE': 'laugh',
                     'ABATTOIR': 'killing place',
                     'VITTLES': 'food',
                     'SKIRMISH': 'fight'}
# FINISH.

    # package together and return results
    # ADD REMAINING MEASURES LATER
    results = {'promis_anx': promis_anx,
               'promis_ang': promis_ang,
               'promis_dep': promis_dep,
               'promis_pos': promis_pos,
               'bite': bite}
               #'ders': ders,
               #'dass': dass,
               #'promis_sat': promis_sat,
               #'vocab': vocab}
    return results