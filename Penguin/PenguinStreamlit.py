import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Penguin prediction App      

This app predicts the ***Palmer Penguin*** Species

Data obtained from the [palmerpenguins library](https://github.com/allisonhorst/palmerpenguins) in R by Allison Horst.
""")


st.sidebar.header("User Input Features")

st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
""")

# Collects user input into the dataframe
upload_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if upload_file is not None:
    input_df = pd.read_csv(upload_file)
else:
    def user_input_features():
        island = st.sidebar.selectbox("Island", ("Biscoe", "Dream", "Torgersen"))
        sex = st.sidebar.selectbox("Sex", ("male", "female"))
        bill_length = st.sidebar.slider("Bill Length (mm)", 39.1, 59.6, 40.0)
        bill_depth = st.sidebar.slider("Bill Depth(mm)", 13.1, 21.5, 17.0)
        flipper_length = st.sidebar.slider("Flipper Length (mm)", 172.0, 231.0, 190.0)
        body_mass = st.sidebar.slider("Body Mass (g)", 2700.0, 6300.0, 4000.0)
        data = {"island" : island,
                "sex" : sex,
                "bill_length_mm" : bill_length,
                "bill_depth_mm" : bill_depth,
                "flipper_length_mm" : flipper_length,
                "body_mass_g" : body_mass,
                }
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

# Encoding the raw data
penguins_raw = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/code/master/streamlit/part3/penguins_cleaned.csv")
penguins = penguins_raw.drop(columns=["species"])

df = pd.concat([input_df, penguins], axis=0)

# Encoding the ordinal features
encode = ['sex', 'island']

# Encoding the variables
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]


# Selecting only the first row
df = df[:1] 


# Displays the user input features
st.subheader("User Input Features")

if upload_file is not None:
    st.write(df)
else:
    st.write("Awaiting CSV file to be uploaded")
    st.write(df)

# Loading the random forest model
load_clf = pickle.load(open('C:/Users/hoybr/sessionworkspace/Streamlit/Penguin/penguins_clf.pkl', 'rb'))

# Applying model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)


# Displaying the predicted penguin
st.subheader("Prediction")
penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
st.write(penguins_species[prediction])

# Creating a species mapping
species_mapping = {0:'Adelie', 1:'Chinstrap', 2:'Gentoo'}

# Creating a dataframe
prediction_df = pd.DataFrame(
    prediction_proba,
    columns=[species_mapping[i] for i in range(len(species_mapping))]
)

st.subheader("Prediction Probability")
st.write(prediction_df)
