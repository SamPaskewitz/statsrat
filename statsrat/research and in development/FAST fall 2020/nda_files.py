

# ***** IMPORT DATA FILE *****
old_ident = pd.read_csv('summary_fall2020 (ml with max_time = 30).csv')['ident']
summary_fall2020 = pd.read_csv('summary_fall2020.csv')
summary_fall2020 = summary_fall2020.loc[summary_fall2020['ident'].isin(old_ident)]
print(summary_fall2020)
n = summary_fall2020.shape[0] # Total number of participants in the data set.

# ***** CALCULATE AGE IN MONTHS *****
age_code_inlab = pd.Series({'18-24': 12*21, # Take the middle of each bracket and multiply by 12 months/year.
                          '25-34': 12*29.5, 
                          '35-44': 12*39.5, 
                          '45-54': 12*49.4, 
                          '55-64': 12*59.5, 
                          '65-74': 12*69.5, 
                          '75+': 12*75})

age_code_online = pd.Series({'18 years old': 12*18, '19 years old': 12*19, '20 years old': 12*20, '21 years old': 12*21, '22 years old': 12*22, '23 years old': 12*23, '24 years or older': 12*24})

age_months = np.zeros(n)
for i in range(n):
    if summary_fall2020['source'].iloc[i] == 'inlab':
        age_months[i] = age_code_inlab[summary_fall2020['age'].iloc[i]]
    else:
        age_months[i] = age_code_online[summary_fall2020['age'].iloc[i]]

# ***** DEFINE RELEVANT DICTIONARIES *****

sex_code = pd.Series({'Female': 'F', 'Male': 'M', 'Intersex': 'O', 'Prefer not to answer': 'NR'})

platform_code = pd.Series({'inlab': 'Psychopy', 'online': 'Testable'})

# 1=Unemployed; 2=Unemployed Stay-at-home parent; 3=Part-Time Student; 4=Full-Time Student; 5=Disability/Unemployed; 6=Disability/Part-Time Employment; 7=Employed Part Time; 8=Employed Full Time; 9=Retired/Part-Time Employment; 10=Retired/Unemployed.; 11 = Student; 12 = Caring for children, elders and house; 13 = Unemployed, volunteer work
employment_code = pd.Series({'Employed full-time': 8,
                           'Employed part-time': 7,
                           'Disabled (not working because of permanent or temporary disability)': 5,
                           'Homemaker': 12,
                           'Full-time student': 4,
                           'Retired': 9, 
                           'Seeking opportunities': 1,
                           'Prefer not to answer': ''}) # There's no numeric code given for this response, so I'll leave it blank.

# ***** CREATE DATAFRAMES FOR NDA SUBMISSION *****

fast = pd.DataFrame({'subjectkey': summary_fall2020['guid'], # GUID
                     'src_subject_id': summary_fall2020['ident'], # Lab ID
                     'interview_date': summary_fall2020['date'], # Date of Test MM/DD/YYYY
                     'interview_age': age_months, # Age (months)
                     'sex': sex_code[summary_fall2020['sex']], # M;F; O; NR
                     'taskname': n*['facial_affect_salience_task'],
                     'platform': platform_code[summary_fall2020['source']], # Software platform used to display the task.
                     'setting': summary_fall2020['source'], # Setting of the task, e.g. online at home, in lab, online in lab.
                     'data_file': n*[''], # Output data file from task. LEAVE BLANK FOR NOW.
                     'data_file_type': n*['']}) # Type of data file.  LEAVE BLANK FOR NOW.

ndar_subject01 = pd.DataFrame({'subjectkey': summary_fall2020['guid'],
                               'src_subject_id': summary_fall2020['ident'],
                               'interview_date': summary_fall2020['date'].values,
                               'interview_age': age_months, # Age in months at the time of the interview/test/sampling/imaging.
                               'sex': sex_code[summary_fall2020['sex']], # M = Male; F = Female; O=Other; NR = Not reported; Gender (if mismatch with natal sex, report Other or Not Reported)
                               'race': summary_fall2020['race'], # American Indian/Alaska Native; Asian; Hawaiian or Pacific Islander; Black or African American; White; More than one race; Unknown or not reported
                               'ethnic_group': summary_fall2020['latinx'], # demo_ethnicity,demqrea1,ethnicity_dem,ethnicityid,hispanic
                               'phenotype': summary_fall2020['source'], # undergrad/online *** CHECK THIS ***
                               'employ_status': employment_code[summary_fall2020['employment']]})

prang01 = pd.DataFrame({'subjectkey': summary_fall2020['guid'],
                        'src_subject_id': summary_fall2020['ident'],
                        'interview_age': age_months,
                        'interview_date': summary_fall2020['date'],
                        'sex': sex_code[summary_fall2020['sex']],
                        'edang03': summary_fall2020['promis_ang_resp1'], # I was irritated more than people knew
                        'edang09': summary_fall2020['promis_ang_resp2'], # I felt angry
                        'edang15': summary_fall2020['promis_ang_resp3'], # I felt like I was ready to explode
                        'edang30': summary_fall2020['promis_ang_resp4'], # I was grouchy
                        'edang35': summary_fall2020['promis_ang_resp5'], # I felt annoyed
                        'anger_rs': summary_fall2020['promis_ang_sum'],
                        'anger_ts': summary_fall2020['promis_ang'],
                        'anger_se': n*[''], # LEAVE BLANK FOR NOW.
                        'anger_theta': n*[''], # LEAVE BLANK FOR NOW.
                        'version_form': n*['PROMIS Item Bank v. 1.1 – Emotional Distress - Anger - Short Form 5a']}) # Form used/assessment name

preda01 = pd.DataFrame({'subjectkey': summary_fall2020['guid'],
                        'src_subject_id': summary_fall2020['ident'],
                        'interview_age': age_months, # Age in months at the time of the interview/test/sampling/imaging.
                        'interview_date': summary_fall2020['date'], # Date on which the interview/genetic test/sampling/imaging/biospecimen was completed. MM/DD/YYYY
                        'sex': sex_code[summary_fall2020['sex']],
                        'edanx01': summary_fall2020['promis_anx_resp1'], # I felt fearful
                        'edanx40': summary_fall2020['promis_anx_resp2'], # I found it hard to focus on anything other than my anxiety
                        'edanx41': summary_fall2020['promis_anx_resp3'], # My worries overwhelmed me
                        'edanx53': summary_fall2020['promis_anx_resp4'], # I felt uneasy
                        'edanx46': summary_fall2020['promis_anx_resp5'], # I felt nervous
                        'edanx07': summary_fall2020['promis_anx_resp6'], # I felt like I needed help for my anxiety
                        'edanx05': summary_fall2020['promis_anx_resp7'], # I felt anxious
                        'edanx54': summary_fall2020['promis_anx_resp8'], # I felt tense
                        'anx_rs': summary_fall2020['promis_anx_sum'], # PROMIS Anxiety raw score
                        'anx_ts': summary_fall2020['promis_anx'], # PROMIS Anxiety T score
                        'anx_se': n*[''], # LEAVE BLANK FOR NOW.
                        'anx_theta': n*[''], # LEAVE BLANK FOR NOW.
                        'version_form': n*['PROMIS Item Bank v1.0 – Emotional Distress – Anxiety – Short Form 8a']}) # Form used/assessment name

print(fast.head())
print(ndar_subject01.head())
print(prang01.head())
print(preda01.head())