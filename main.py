import pandas as pd
import numpy as np
import joblib
import streamlit as st

# title if the web app
st.title("Email Classifier")
# load the model
model = joblib.load("emailclassifier.pkl")
# load the vectorizer
vectorizer = joblib.load("countvectorizer.pkl")
# take the input of email from the user
email = st.text_area("Enter the email here")

# make the prediction button
# make the prediction button
if st.button("predict"):
    if email.strip() == "":
        st.warning("please enter an email message")
    else:
        #convert the email text into vectors 
        email_vectors = vectorizer.transform([email])
        prediction = model.predict(email_vectors)
    # display the result 
    if prediction[0] == 'spam':
        st.error("🚨 This is a SPAM Email!")
    else:
        st.success("this is a ham email")
