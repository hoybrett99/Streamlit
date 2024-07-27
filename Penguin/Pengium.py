import pandas as pd
penguins = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/code/master/streamlit/part3/penguins_cleaned.csv")


# Copying the dataset
df = penguins.copy()

# Target variable
target = 'species'

# Variables that need encoding
encode = ['sex', 'island']

# Creating a loop to encode each of the columns that needs encoding
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]

# Encoding the target variable
target_mapper = {'Adelie':0, 'Chinstrap':1, 'Gentoo':2}

def target_encode(val):
    return target_mapper(val)

df['species'] = df['species'].apply(target_encode)