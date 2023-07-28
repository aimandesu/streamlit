import streamlit as st
import pandas as pd
import joblib
import streamlit.components.v1 as components
import os

# checking accuracy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import mean_squared_error

# Models used, KNN, SVM, and Random Forest
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from PIL import Image


st.title("Chronic Kidney Disease")
st.subheader("What is Chronic Kidney Disease?")

description = " \
                Chronic Kidney Disease (CKD) is a long-term condition characterized by the gradual loss of kidney function over time. \
                The kidneys play a crucial role in filtering waste products and excess fluid from the blood, maintaining electrolyte balance, and producing hormones that regulate blood pressure. \
                CKD occurs when the kidneys are damaged and are unable to perform these functions effectively. \
                The most common causes of CKD are diabetes, high blood pressure, and certain kidney infections or inherited disorders.\
                "
type = "\
        There are two main types of kidney disease - short-term (acute kidney injury) and lifelong (chronic). \
        An acute kidney injury is the temporary loss of kidney function lasting less than three months.\
        It typically has a fast onset, in response to an injury or illness affecting the kidneys, drugs, blockages of the kidney or many other factors.\
        Chronic kidney disease occurs when your kidneys have been damaged in a way that cannot be reversed.\
        To be diagnosed with a chronic kidney disease, the condition will need to have been present for at least three months.\
        "


symptoms = "\
            Symptoms of CKD may include fatigue, loss of appetite, swelling in the hands and feet, frequent urination, and changes in urine color. \
            Early detection and management of CKD are crucial to slow down its progression and prevent further kidney damage. \
            Treatment options may include lifestyle modifications, medication to control underlying conditions, dietary changes, and in advanced cases, dialysis or kidney transplantation.\
            "
dataset_description = "\
            The datasets consist of age, values for 2 = yes, and 1 = no for RedBlood, Specific Gravity, Sugar,  \
            Blood Urea, Serum Creatine, Pottasium, Hemoglobin, White Blood Cell Count, Red Blood Cell Count, Hypertension and result of Classification Chronic Kidney Disease consist. \
            of YES or NO\
            "

variable = "\
            The variable input will be used are all the datasets features except the Age and result of Classification Chronic Kidney Disease,  \
            while the variable output will be result of Chronic Kidney Disease \
            \
            "

st.write(
    f'<p style="text-align: justify; font-size: 20px;">{description}</p>', unsafe_allow_html=True)
st.write(
    f'<p style="text-align: justify; font-size: 20px;">{type}</p>', unsafe_allow_html=True)

st.subheader("Symptoms")
st.write(
    f'<p style="text-align: justify; font-size: 20px;">{symptoms}</p>', unsafe_allow_html=True)

st.subheader("Chronic Disease Kideny Diagnosis")

col1, col2 = st.columns(2)
with col1:
    image = Image.open('images\kidney\kidney1.png')
    newImage = image.resize((600, 600))
    st.image(newImage, caption='Outer')

with col2:
    image = Image.open('images\kidney\kidney2.png')
    newImage = image.resize((600, 600))
    st.image(newImage, caption='Inner')

st.header("Analysis")
st.write(
    f'<p style="text-align: justify; font-size: 20px;">{"In this section, the web application is used to analysis whether the user has Chronic Kidney Disease or not."}</p>', unsafe_allow_html=True)

st.subheader("Dataset description")
st.write(
    f'<p style="text-align: justify; font-size: 20px;">{dataset_description}</p>', unsafe_allow_html=True)

st.subheader("Variable Input / Output")
st.write(
    f'<p style="text-align: justify; font-size: 20px;">{variable}</p>', unsafe_allow_html=True)

st.subheader("Model")
st.write(
    f'<p style="text-align: justify; font-size: 20px;">{"Pick models to use: KNN, Random Forest, SVM."}</p>', unsafe_allow_html=True)


# components.html(
#     """
# <div>
#     <p style="text-align: justify; color: white; font-size: 30px; height: 100%">
#         Lung cancer is a type of cancer that begins in the cells of the lungs.
#         It is one of the most common and deadliest forms of cancer worldwide.
#         The primary function of the lungs is to supply oxygen to the body and remove carbon dioxide during breathing.
#         Lung cancer disrupts this normal function and can spread to other parts of the body through the bloodstream or lymphatic system.
#     </p>
# </div>
#     """,
#     # height=300
# )

# cancerDescription = "Lung cancer is a type of cancer that begins in the cells of the lungs. \
#         It is one of the most common and deadliest forms of cancer worldwide. \
#         The primary function of the lungs is to supply oxygen to the body and remove carbon dioxide during breathing. \
#         Lung cancer disrupts this normal function and can spread to other parts of the body through the bloodstream or lymphatic system."

# st.title(f'<p style="background-color:red;color:#33ff33;font-size:24px;border-radius:2%;">{cancerDescription}</p>', unsafe_allow_html=True)

# def header(thisThing):
#      st.markdown(f'<p style="background-color:red;color:#33ff33;font-size:24px;border-radius:2%;">{thisThing}</p>', unsafe_allow_html=True)

# header("Test")


dataset = pd.read_csv("dataset/ChronicKidneyDisease.csv")

selectModel = st.selectbox("Select model", options=[
                           "", "all", "KNN", "Random Forest", "SVM"])

data_input_training = dataset.drop(columns=["Age", "Classification"])
data_target_training = dataset["Classification"]

st.subheader("Training and Testing")
st.write(
    f'<p style="text-align: justify; font-size: 20px;">{"Training and Testing data will be divided using train test split."}</p>', unsafe_allow_html=True)

if (selectModel != ""):

    # all these training need to be saved in joblib
    if os.path.exists("joblib/tts_indices_ChronicKidneyDisease.joblib"):
        X_train, X_test, y_train, y_test = joblib.load(
            'joblib/tts_indices_ChronicKidneyDisease.joblib')
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            data_input_training, data_target_training, test_size=0.2)
        joblib.dump((X_train, X_test, y_train, y_test),
                    'joblib/tts_indices_ChronicKidneyDisease.joblib')

    st.write("data input training")
    X_train

    st.write("data target training")
    y_train

    st.write("data input testing")
    X_test

    st.write("data target testing")
    y_test

    if selectModel == "all":
        st.write("KNN")
        neighbors = [1, 5, 10, 15]

        for selectNeigbors in neighbors:
            knn = KNeighborsClassifier(n_neighbors=selectNeigbors)
            knn.fit(X_train, y_train)

            st.write("prediction knn:", selectNeigbors)
            prediction = knn.predict(X_test)
            prediction

            st.write("accuracy knn: ", selectNeigbors)
            accuracy = accuracy_score(y_test, prediction)
            accuracy

        st.write("RF")
        estimators = [100, 500, 1000]

        for selectEstimator in estimators:
            rf = RandomForestClassifier(n_estimators=selectEstimator)
            rf.fit(X_train, y_train)

            st.write("prediction RF:", selectEstimator)
            prediction = rf.predict(X_test)
            prediction

            st.write("accuracy RF: ", selectEstimator)
            accuracy = accuracy_score(y_test, prediction)
            accuracy

        st.write("SVM")
        kernel = ["linear", "rbf", "poly", "sigmoid"]

        for selectKernel in kernel:
            svm = SVC(kernel=selectKernel)
            svm.fit(X_train, y_train)

            st.write("prediction SVM:", selectKernel)
            prediction = svm.predict(X_test)
            prediction

            st.write("accuracy SVM: ", selectKernel)
            accuracy = accuracy_score(y_test, prediction)
            accuracy

        st.write("Go back at top? click [here](#model)")

    elif selectModel != "all":

        question = [
            "What is your Blood Pressure?",
            "State your Specific Gravity",
            "What is your Albumin Level?",
            "What is your Sugar Level?",
            "Do you have too many Blood Cell?",
            "State your Blood Urea",
            "State your serum creatine",
            "State your Sodium",
            "State your pottasium",
            "State your Hemoglobin",
            "State how many your White Blood Cell Count",
            "State how many your Red Blood Cell Count",
            "Do you have Hypertension?",
        ]

        st.subheader("Question")
        st.write(
            f'<p style="text-align: justify; font-size: 20px;">{"Question time. Please answer as honestly as possible."}</p>', unsafe_allow_html=True)

        for index, element in enumerate(question):

            # checking if its yes or no question
            if (element == "What is your Blood Pressure?"):
                answer = st.slider(element, 0, 200)
            elif (element == "What is your Albumin Level?"):
                answer = st.slider(element, 0, 4)
            elif (element == "What is your Sugar Level?"):
                answer = st.slider(element, 0, 5)
            elif (element == "State your Blood Urea"):
                answer = st.slider(element, 0, 200)
            elif (element == "Do you have Hypertension?"):
                answer = st.selectbox(element, options=["Yes", "No"])
            elif (element == "Do you have too many Blood Cell?"):
                answer = st.selectbox(element, options=["Yes", "No"])
            else:
                answer = st.number_input(element)

            # if input either yes or no
            if answer == "Yes":
                answer = 1
            elif answer == "No":
                answer = 0

            # change answer to the array
            question[index] = answer

        if (selectModel == "KNN"):
            container_2 = st.empty()
            value = container_2.button('Start Analysis with KNN')

            st.write("KNN")
            # selectNeigbors = st.select_slider("select neighbors to use:", options=[1, 5, 10, 15 ], disabled=value)
            selectNeigbors = st.selectbox("select neighbors to use:", options=[
                1, 5, 10, 15], disabled=value)

            if value:
                container_2.empty()
                container_2.button('End Session KNN')

            if value:
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

                # st.write("testing if it is accurate or not..")
                # question
                test = knn.predict([question, question])
                question

                st.subheader("Result")
                result = test[0]
                if (result == 1):
                    st.write(
                        f'<p style="text-align: justify; font-size: 20px;">{"Unfortunately, it appears that you have Chronic Kidney Disease."}</p>', unsafe_allow_html=True)
                else:
                    st.write(
                        f'<p style="text-align: justify; font-size: 20px;">{"Great! it does not seem that you have Chronic Kidney Disease"}</p>', unsafe_allow_html=True)

                # testJoblib = loaded_model.predict([question, question])
                # testJoblib

                # bro = loaded_model.predict(X_test)
                # bro

        elif (selectModel == "Random Forest"):
            container_2 = st.empty()
            value = container_2.button('Start Analysis with RF')

            st.write("Random Forest")
            selectEstimator = st.selectbox("select estimators to use:", options=[
                100, 500, 1000,], disabled=value)

            if value:
                container_2.empty()
                container_2.button('End Session RF')

            if value:
                rf = RandomForestClassifier(n_estimators=selectEstimator)
                rf.fit(X_train, y_train)

                st.write("prediction RF:", selectEstimator)
                prediction = rf.predict(X_test)
                prediction

                st.write("accuracy RF: ", selectEstimator)
                accuracy = accuracy_score(y_test, prediction)
                accuracy

                test = rf.predict([question, question])
                question

                st.subheader("Result")
                result = test[0]
                if (result == 1):
                    st.write(
                        f'<p style="text-align: justify; font-size: 20px;">{"Unfortunately, it appears that you have Chronic Kidney Disease."}</p>', unsafe_allow_html=True)
                else:
                    st.write(
                        f'<p style="text-align: justify; font-size: 20px;">{"Great! it does not seem that you have Chronic Kidney Disease"}</p>', unsafe_allow_html=True)

        elif (selectModel == "SVM"):
            container_2 = st.empty()
            value = container_2.button('Start Analysis with SVM')

            st.write("Support Vector Machine")
            selectKernel = st.selectbox("select kernels to use:", options=[
                "linear", "rbf", "poly", "sigmoid",], disabled=value)

            if value:
                container_2.empty()
                container_2.button('End Session SVM')

            if value:
                svm = SVC(kernel=selectKernel)
                svm.fit(X_train, y_train)

                st.write("prediction SVM:", selectKernel)
                prediction = svm.predict(X_test)
                prediction

                st.write("accuracy SVM: ", selectKernel)
                accuracy = accuracy_score(y_test, prediction)
                accuracy

                test = svm.predict([question, question])
                question

                st.subheader("Result")
                result = test[0]
                if (result == 1):
                    st.write(
                        f'<p style="text-align: justify; font-size: 20px;">{"Unfortunately, it appears that you have Chronic Kidney Disease."}</p>', unsafe_allow_html=True)
                else:
                    st.write(
                        f'<p style="text-align: justify; font-size: 20px;">{"Great! it does not seem that you have Chronic Kidney Disease"}</p>', unsafe_allow_html=True)

        if (value):
            st.subheader("Analysis Ends")
            st.write("Want to try other model? click [here](#model)")
            st.write("Want to alter your question? click [here](#question)")

else:
    st.warning("Please select model at model section to proceed.")
