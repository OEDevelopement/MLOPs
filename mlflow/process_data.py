import pandas as pd

df = pd.read_csv('data/raw/adult.csv')

# simplifying occupation
df['occupation'] = df['occupation'].apply(lambda wert: 
    'Simple Services' if wert in ['Transport-moving', 'Handlers-cleaners', 'Priv-house-serv', 'Machine-op-inspct', 'Other-service'] else 
    'Public Safety' if wert in ['Protective-serv', 'Armed-Forces'] else 
    'Specialized Services' if wert in ['Craft-repair', 'Tech-support'] else 
    'Professional' if wert in ['Prof-specialty', 'Farming-fishing'] else 
    'Management' if wert == 'Exec-managerial' else 
    'Administrative' if wert == 'Adm-clerical' else 
    wert
)

# simplifying workclass    
df['workclass'] = df['workclass'].apply(lambda wert: 
    'Government' if wert in ['Local-gov', 'State-gov', 'Federal-gov'] else 
    'Self Employed' if wert in ['Self-emp-not-inc', 'Self-emp-inc'] else 
    'Unemployed' if wert in ['Without-pay', 'Never-worked'] else 
    wert
)

# simplifying marital-status    
df['marital-status'] = df['marital-status'].apply(lambda wert: 
    'Widowed/Separated' if wert in ['Divorced', 'Separated', 'Widowed'] else 
    'Married' if wert in ['Married-civ-spouse', 'Married-spouse-absent', 'Married-AF-spouse'] else 
    wert
)

# simplifying relationship    
df['relationship'] = df['relationship'].apply(lambda wert: 
    'Shared Housing' if wert in ['Not-in-family', 'Other-relative'] else 
    'Child' if wert == 'Own-child' else 
    'Single' if wert == 'Unmarried' else 
    wert
)

# binarizing race
df['is_White'] = df['race'].apply(lambda wert: 
    1 if wert == 'White' else 0
)

# binarizing gender
df['is_Male'] = df['gender'].apply(lambda wert: 
    1 if wert == 'Male' else 0
)

# binarizing income
df['income >50K'] = df['income'].apply(lambda wert: 
    1 if wert == '>50K' else 0
)

# binarizing native-country
df['from_USA'] = df['native-country'].apply(lambda wert: 
    1 if wert == 'United-States' else 0
)

# binarizing gained-capitaö
df['capital-net'] = df['capital-gain'] - df['capital-loss']
df['gained-capital'] = df['capital-net'].apply(lambda wert: 
    1 if wert <= 0 else 0
)

# drop Fragezeichen    
df = df[~df.isin(['?']).any(axis=1)].copy()

# drop irrelevant columns
df.drop(columns=['education', 'fnlwgt', 'native-country', 'capital-gain', 'capital-loss', 'gender', 'income', 
                 'race', 'capital-net'], inplace=True)

df.to_csv('data/processed/processed_data.csv', index=False)