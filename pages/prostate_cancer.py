import streamlit as st
import pandas as pd
import joblib
import streamlit.components.v1 as components
import os

#checking accuracy 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import mean_squared_error

#Models used, KNN, SVM, and Random Forest
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from PIL import Image



st.title("LUNG CANCER")
st.subheader("What is lung cancer?")

description =   " \
                Lung cancer is a type of cancer that begins in the cells of the lungs. \
                It is one of the most common and deadliest forms of cancer worldwide. \
                The primary function of the lungs is to supply oxygen to the body and remove carbon dioxide during breathing. \
                Lung cancer disrupts this normal function and can spread to other parts of the body through the bloodstream or lymphatic system.\
                "

type =  "\
        There are two main types of lung cancer: non-small cell lung cancer (NSCLC) and small cell lung cancer (SCLC). \
        NSCLC is the most common type, accounting for about 80-85% of all lung cancers. It includes subtypes such \
        as adenocarcinoma, squamous cell carcinoma, and large cell carcinoma. SCLC is a more aggressive type, typically \
        growing and spreading at a faster rate.\
        "

symptoms =  "\
            The symptoms of lung cancer may vary depending on the stage and type of cancer. Common symptoms include persistent cough, \
            chest pain, shortness of breath, coughing up blood, fatigue, unexplained weight loss, and recurrent respiratory infections. \
            Early detection of lung cancer is crucial for better treatment outcomes.\
            "
dataset_description = "\
            The datasets consist of gender, age, values for 2 = yes, and 1 = no for smoking, yellow fingers, anxiety, peer pressure, chronic disease,  \
            fatigue, allergy, wheezing, alcohol consumption, coughing, shortness of breath, swallow dificulity, chest pain, and result of lung cancer consist. \
            of YES or NO\
            "

variable = "\
            The variable input will be used are all the datasets features except the gender and result of lung cancer,  \
            while the variable output will be result of lung cancer\
            \
            "

st.write(f'<p style="text-align: justify; font-size: 20px;">{description}</p>', unsafe_allow_html=True)
st.write(f'<p style="text-align: justify; font-size: 20px;">{type}</p>', unsafe_allow_html=True)

st.subheader("Symptoms")
st.write(f'<p style="text-align: justify; font-size: 20px;">{symptoms}</p>', unsafe_allow_html=True)

st.subheader("Lung Diagnosis")

col1, col2 = st.columns(2)
with col1:
    image = Image.open('images\lung\lung_cancer_1.jpg')
    newImage = image.resize((600, 600))
    st.image(newImage, caption='Outer')

with col2:
    image = Image.open('images\lung\lung_cancer_2.jpg')
    newImage = image.resize((600, 600))
    st.image(newImage, caption='Inner')

st.header("Analysis")
st.write(f'<p style="text-align: justify; font-size: 20px;">{"In this section, the web application is used to analysis whether the user has lung cancer or not."}</p>', unsafe_allow_html=True)

st.subheader("Dataset description")
st.write(f'<p style="text-align: justify; font-size: 20px;">{dataset_description}</p>', unsafe_allow_html=True)

st.subheader("Variable Input / Output")
st.write(f'<p style="text-align: justify; font-size: 20px;">{variable}</p>', unsafe_allow_html=True)

st.subheader("Model")
st.write(f'<p style="text-align: justify; font-size: 20px;">{"Pick models to use: KNN, Random Forest, SVM."}</p>', unsafe_allow_html=True)


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


dataset = pd.read_csv("dataset/survey_lung_cancer.csv")

selectModel = st.selectbox("Select model", options=["", "KNN", "Random Forest", "SVM"])

data_input_training = dataset.drop(columns = ["LUNG_CANCER", "GENDER"])
data_target_training = dataset["LUNG_CANCER"]

st.subheader("Training and Testing")
st.write(f'<p style="text-align: justify; font-size: 20px;">{"Training and Testing data will be divided using train test split."}</p>', unsafe_allow_html=True)

if(selectModel !=""):

    #all these training need to be saved in joblib
    if os.path.exists("joblib/tts_indices_lung_cancer.joblib"):
        X_train, X_test, y_train, y_test = joblib.load('joblib/tts_indices_lung_cancer.joblib')
    else:
        X_train, X_test, y_train, y_test = train_test_split(data_input_training, data_target_training, test_size=0.2)
        joblib.dump((X_train, X_test, y_train, y_test), 'joblib/tts_indices_lung_cancer.joblib')

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

    st.subheader("Question")
    st.write(f'<p style="text-align: justify; font-size: 20px;">{"Question time. Please answer as honestly as possible."}</p>', unsafe_allow_html=True)

    for index, element in enumerate(question):

        #checking if its yes or no question
        if(element == "What is your age?"):
            answer = st.slider(element, 0, 100)
            # answer = st.slider("element", min_value=0.0, max_value=0.1, step=0.01)
            # answer = st.number_input(element)
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
        value = container_2.button('Start Analysis with KNN')

        st.write("KNN")
        # selectNeigbors = st.select_slider("select neighbors to use:", options=[1, 5, 10, 15 ], disabled=value)
        selectNeigbors = st.selectbox("select neighbors to use:", options=[1, 5, 10, 15 ], disabled=value)

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
           
            st.subheader("Result")
            result = test[0]
            if(result == "YES"):
                st.write(f'<p style="text-align: justify; font-size: 20px;">{"Unfortunately, it appears that you have lung cancer."}</p>', unsafe_allow_html=True)
            else:
                st.write(f'<p style="text-align: justify; font-size: 20px;">{"Great! it does not seem that you have lung cancer"}</p>', unsafe_allow_html=True)

            # testJoblib = loaded_model.predict([question, question])
            # testJoblib

            # bro = loaded_model.predict(X_test)
            # bro

    elif(selectModel == "Random Forest"):
        container_2 = st.empty()
        value = container_2.button('Start Analysis with RF')

        st.write("Random Forest")
        selectEstimator = st.selectbox("select estimators to use:", options=[100, 500, 1000,], disabled=value)

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

            st.subheader("Result")
            result = test[0]
            if(result == "YES"):
                st.write(f'<p style="text-align: justify; font-size: 20px;">{"Unfortunately, it appears that you have lung cancer."}</p>', unsafe_allow_html=True)
            else:
                st.write(f'<p style="text-align: justify; font-size: 20px;">{"Great! it does not seem that you have lung cancer"}</p>', unsafe_allow_html=True)

    elif(selectModel == "SVM"):
        container_2 = st.empty()
        value = container_2.button('Start Analysis with SVM')

        st.write("Support Vector Machine")
        selectKernel = st.selectbox("select kernels to use:", options=["linear", "rbf", "poly", "sigmoid",], disabled=value)

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

            st.subheader("Result")
            result = test[0]
            if(result == "YES"):
                st.write(f'<p style="text-align: justify; font-size: 20px;">{"Unfortunately, it appears that you have lung cancer."}</p>', unsafe_allow_html=True)
            else:
                st.write(f'<p style="text-align: justify; font-size: 20px;">{"Great! it does not seem that you have lung cancer"}</p>', unsafe_allow_html=True)

    if(value):
        st.subheader("Analysis Ends")
        st.write("Want to try other model? click [here](#model)")
        st.write("Want to alter your question? click [here](#question)")

else:
    st.warning("Please select model at model section to proceed.")


