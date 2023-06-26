import streamlit as st
import pandas as pd
import joblib

#checking accuracy 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import mean_squared_error

#Models used, KNN, SVM, and Random Forest
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor



st.title("LUNG CANCER")
st.header("description")
st.write("explain stuff...")

dataset = pd.read_csv("dataset/survey_lung_cancer.csv")

st.write("Select Model")

selectModel = st.selectbox("Select model", options=["", "KNN", "Random Forest", "SVM"])

data_input_training = dataset.drop(columns = ["LUNG_CANCER", "GENDER"])
data_target_training = dataset["LUNG_CANCER"]

st.subheader("Training and Testing data will be divided using train test split")

if(selectModel !=""):

    #all these training need to be saved in joblib
    X_train, X_test, y_train, y_test = train_test_split(data_input_training, data_target_training, test_size=0.2)

    st.write("data input training")
    X_train

    st.write("data target training")
    y_train

    st.write("data input testing")
    X_test

    st.write("data target testing")
    y_test
    
    question = [
        "What is your age?", "Do you smoke?", "Are your fingers yellow?", "Are you having anxiety frequently?", "Do you have pressure from your peers?", 
        "Do you have any chronic disease?","Do you have fatigue?", "Are you allergy to anything?", "Do you alwasy wheezing a lot?", "Do you take alcohol?", 
        "Do you coughing a lot these days?", "Are you having shortness of breath?", "Do you have swallowing difficulity?", "Do you have chest pain sometimes?",
        ]

    for index, element in enumerate(question):

        #checking if its yes or no question
        if(element == "What is your age?"):
            answer = st.slider(element, 0, 100)
        else:
            answer = st.selectbox(element, options=["Yes", "No"])

        #if input either yes or no
        if answer == "Yes":
            answer = 2;
        elif answer == "No":
            answer = 1;
        
        #change answer to the array
        question[index] = answer

    if(selectModel == "KNN"):
        container_2 = st.empty()
        value = container_2.button('Start')

        st.write("KNN")
        selectNeigbors = st.select_slider("select neighbors to use:", options=[0, 1, 5, 10, 15 ], disabled=value)

        if value:
          container_2.empty()
          container_2.button('End')

        if(value):
            knn = KNeighborsClassifier(n_neighbors=selectNeigbors)
            knn.fit(X_train, y_train)

            st.write("prediction knn:", selectNeigbors)
            prediction = knn.predict(X_test)
            prediction

            st.write("accuracy knn: ", selectNeigbors)
            accuracy = accuracy_score(y_test, prediction)
            accuracy

            # joblib.dump(knn, 'knn_model.joblib')
            # loaded_model = joblib.load('knn_model.joblib')

            st.write("testing if it is accurate or not..")
            question 
            test = knn.predict([question, question])
            test

            # testJoblib = loaded_model.predict([question, question])
            # testJoblib

            # bro = loaded_model.predict(X_test)
            # bro

        else:
            st.write("neighbors must not be 0")

    elif(selectModel == "Random Forest"):
        st.write("Hey this is random forest")
    elif(selectModel == "SVM"):
        st.write("Hey this is SVM")

else:
    st.write("Please select model at above section")


