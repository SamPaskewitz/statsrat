
# Later, add a function to process Psychopy survey data.

def process_testable_surveys(df):
    # set up variables
    anx, ang, dep, pos, bite, ders, dass, satsf, vocab = [], [], [], [], [], [], [], []    

    # PROMIS anxiety
    index = df.responseRows == 'I felt fearful; I found it hard to focus on anything other than my anxiety; My worries overwhelmed me; I felt uneasy; I felt nervous; I felt like I needed help for my anxiety; I felt anxious; I felt tense'
    responses = np.array(df['responseCode'].loc[index].split(separator = '_'), dtype = 'float')
    anx += [responses.sum()]

    # PROMIS anger
    index = df.responseRows == 'I was irritated more than people knew; I felt angry; I felt like I was ready to explode; I was grouchy; I felt annoyed'
    responses = np.array(df['responseCode'].loc[index].split(separator = '_'), dtype = 'float')
    ang += [responses.sum()]
    
    # PROMIS depression
    index = df.responseRows == 'I felt worthless; I felt helpless; I felt depressed; I felt hopeless; I felt like a failure; I felt unhappy; I felt that I had nothing to look forward to; I felt that nothing could cheer me up'
    responses = np.array(df['responseCode'].loc[index].split(separator = '_'), dtype = 'float')
    dep += [responses.sum()]

    # PROMIS positive affect
    index = df.responseRows == 'I felt cheerful; I felt attentive; I felt delighted; I felt happy; I felt joyful; I felt enthusiastic; I felt determined; I felt interested; I was thinking creatively; I liked myself; I felt peaceful; I felt good-natured; I felt useful; I felt understood; I felt content'
    responses = np.array(df['responseCode'].loc[index].split(separator = '_'), dtype = 'float')
    pos += [responses.sum()] 

    # BITE
    index = df.responseRows == 'I have been grumpy; I have been feeling like I might snap; Other people have been getting on my nerves; Things have been bothering me more than they normally do; I have been feeling irritable'
    responses = np.array(df['responseCode'].loc[index].split(separator = '_'), dtype = 'float')
    bite += [responses.sum()] 

    # DERS-18
    # FIX BULLSHIT CHARACTERS
    index0 = df.responseRows == "I pay attention to how I feel.; I have no idea how I am feeling.; I have difficulty making sense out of my feelings.; I am attentive to my feelings.; I am confused about how I feel.; When I‚Äôm upset, I acknowledge my emotions."
    index1 = "When I‚Äôm upset, I become embarrassed for feeling that way.; When I‚Äôm upset, I have difficulty getting work done.; When I‚Äôm upset, I become out of control.; When I'm upset, I believe that I will remain that way for a long time.; When I'm upset, I believe that I'll end up feeling very depressed.; When I'm upset, I have difficulty focusing on other things."
    index2 = "When I'm upset, I feel ashamed with myself for feeling that way.; When I'm upset, I feel guilty for feeling that way.; When I'm upset, I have difficulty concentrating.; When I'm upset, I have difficulty controlling my behaviors.; When I'm upset, I believe that wallowing in it is all I can do.; When I'm upset, I lose control over my behaviors."
    # FINISH.
    
    # DASS-21

    # PROMIS satisfaction with social roles and activities

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