import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Simple Iris Flower Prediction App
         
This app predicts the Iris flower type!
""")

st.sidebar.header("User Input Parameters")

def user_input_features():

    # Creating the sidebar - first value is min, second is max and third is the chosen value
    sepal_length = st.sidebar.slider("Sepal Length", 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider("Sepal Width", 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider("Petal Length", 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider("Petal Width", 0.1, 2.5, 0.2)

    # Dataframe
    data = {"sepal_length": sepal_length,
            "sepal_width": sepal_width,
            "petal_length": petal_length,
            "petal_width": petal_width}
    
    features = pd.DataFrame(data, index=[0])
    return(features)

# Assigned to variable
df = user_input_features()

st.subheader('User Input Parameters')
st.write(df)

# Getting the iris dataset
iris = datasets.load_iris()

# Splitting the variables
X = iris.data
y = iris.target

# Random forest classifier
clf = RandomForestClassifier()

# Fitting the model
clf.fit(X, y)

# Predictions
prediction = clf.predict(df)

# Prediction probability
prediction_proba = clf.predict_proba(df)

# Gives you the class labels of the iris dataset
st.subheader("Class labels and their corresponding index number")
st.write(iris.target_names)

# Prints out the prediction
st.subheader("Prediction")
st.write(iris.target_names[prediction])

# Prediction probability
st.subheader("Prediction Probability")
st.write(prediction_proba)


