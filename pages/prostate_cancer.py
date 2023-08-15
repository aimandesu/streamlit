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


st.title("PROSTATE CANCER")
st.subheader("What is prostate cancer?")

description = " \
                Prostate cancer is a type of cancer that develops in the prostate gland, which is a small walnut-shaped gland located in the male reproductive system. \
                It is one of the most common cancers in men.  \
                Prostate cancer occurs when abnormal cells in the prostate gland grow uncontrollably, forming a tumor that can potentially spread to other parts of the body if not 			detected and treated early.  \
                Regular screenings, such as prostate-specific antigen (PSA) tests and digital rectal exams (DREs), are essential for early detection and improved treatment outcomes.\
                "

type = "\
        There is only one type of prostate cancer, which is adenocarcinoma. \
        Adenocarcinoma is the most common form of prostate cancer, accounting for nearly all cases.\
        "

symptoms = "\
            The symptoms of lung cancer may vary depending on the stage and type of cancer. Common symptoms include persistent cough, \
            chest pain, shortness of breath, coughing up blood, fatigue, unexplained weight loss, and recurrent respiratory infections. \
            Early detection of lung cancer is crucial for better treatment outcomes.\
            "
dataset_description = "\
            The dataset consist of id,radius (1-30), texture (1-30), perimeter (1-200), area (1-2000), smoothness, compactness, \
            symmetry, fractal dimension (0-1) and diagnosis result (M = Malignant, B = Benign).\
            "

variable = "\
            The variable input will be used are all the datasets features except the id and diagnosis result,  \
            while the diagnosis will be result of prostate cancer\
            \
            "

st.write(
    f'<p style="text-align: justify; font-size: 20px;">{description}</p>', unsafe_allow_html=True)
st.write(
    f'<p style="text-align: justify; font-size: 20px;">{type}</p>', unsafe_allow_html=True)

st.subheader("Symptoms")
st.write(
    f'<p style="text-align: justify; font-size: 20px;">{symptoms}</p>', unsafe_allow_html=True)

st.subheader("Prostate Diagnosis")

col1, col2 = st.columns(2)
with col1:
    image = Image.open('images\prostate\prostate_cancer_1.png')
    newImage = image.resize((600, 600))
    st.image(newImage, caption='Front View')

with col2:
    image = Image.open('images\prostate\prostate_cancer_2.png')
    newImage = image.resize((600, 600))
    st.image(newImage, caption='Side View')

st.header("Analysis")
st.write(
    f'<p style="text-align: justify; font-size: 20px;">{"In this section, the web application is used to analysis whether the user has lung cancer or not."}</p>', unsafe_allow_html=True)

st.subheader("Dataset description")
st.write(
    f'<p style="text-align: justify; font-size: 20px;">{dataset_description}</p>', unsafe_allow_html=True)

st.subheader("Variable Input / Output")
st.write(
    f'<p style="text-align: justify; font-size: 20px;">{variable}</p>', unsafe_allow_html=True)

st.subheader("Model")
st.write(
    f'<p style="text-align: justify; font-size: 20px;">{"Pick models to use: KNN, Random Forest, SVM."}</p>', unsafe_allow_html=True)


dataset = pd.read_csv("dataset/prostate.csv")

selectModel = st.selectbox("Select model", options=[
                           "", "all", "KNN", "Random Forest", "SVM"])

data_input_training = dataset.drop(columns=['id', 'diagnosis_result'])
data_target_training = dataset['diagnosis_result']

st.subheader("Training and Testing")
st.write(
    f'<p style="text-align: justify; font-size: 20px;">{"Training and Testing data will be divided using train test split."}</p>', unsafe_allow_html=True)

if (selectModel != ""):

    X_train, X_test, y_train, y_test = train_test_split(
        data_input_training, data_target_training, test_size=0.2)

    # if os.path.exists("joblib/tts_indices_prostate_cancer.joblib"):
    #     X_train, X_test, y_train, y_test = joblib.load(
    #         'joblib/tts_indices_prostate_cancer.joblib')
    # else:
    #     X_train, X_test, y_train, y_test = train_test_split(
    #         data_input_training, data_target_training, test_size=0.2)
    #     joblib.dump((X_train, X_test, y_train, y_test),
    #                 'joblib/tts_indices_prostate_cancer.joblib')

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
            "What is your prostate radius?", "What is the texture number?", "What is the perimeter(cm) of your prostate?",
            "What is the area(cm) of your prostate?", "Smoothness of your prostate (0 = rough,1 = smooth)?",
            "Compactness of your prostate (0 = not compact,1 = compact)?",
            "Symmetry of your prostate (0 = symmetry,1 = not symmetry)?", "Fractal Dimension of your prostate",
        ]

        st.subheader("Question")
        st.write(
            f'<p style="text-align: justify; font-size: 20px;">{"Question time. Please answer as honestly as possible."}</p>', unsafe_allow_html=True)

        for index, element in enumerate(question):

            if (element == "What is your prostate radius?"):
                answer = st.slider(element, 0, 30)
            elif (element == "What is the texture number?"):
                answer = st.slider(element, 0, 30)
            elif (element == "What is the perimeter(cm) of your prostate?"):
                answer = st.slider(element, 0, 200)
            elif (element == "What is the area(cm) of your prostate?"):
                answer = st.slider(element, 0, 2000)
            else:
                st.number_input(element)

            question[index] = answer

        if (selectModel == "KNN"):
            container_2 = st.empty()
            value = container_2.button('Start Analysis with KNN')

            st.write("KNN")

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

                test = knn.predict([question, question])

                st.subheader("Result")
                result = test[0]
                if (result == "M"):
                    st.write(
                        f'<p style="text-align: justify; font-size: 20px;">{"It appears that you have Malignant prostate cancer."}</p>', unsafe_allow_html=True)
                else:
                    st.write(
                        f'<p style="text-align: justify; font-size: 20px;">{"It appears that you have Benign prostate cancer."}</p>', unsafe_allow_html=True)

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

                st.subheader("Result")
                result = test[0]
                if (result == "M"):
                    st.write(
                        f'<p style="text-align: justify; font-size: 20px;">{"It appears that you have Malignant prostate cancer."}</p>', unsafe_allow_html=True)
                else:
                    st.write(
                        f'<p style="text-align: justify; font-size: 20px;">{"It appears that you have Benign prostate cancer."}</p>', unsafe_allow_html=True)

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

                st.subheader("Result")
                result = test[0]
                if (result == "M"):
                    st.write(
                        f'<p style="text-align: justify; font-size: 20px;">{"It appears that you have Malignant prostate cancer."}</p>', unsafe_allow_html=True)
                else:
                    st.write(
                        f'<p style="text-align: justify; font-size: 20px;">{"It appears that you have Benign prostate cancer."}</p>', unsafe_allow_html=True)

        if (value):
            st.subheader("Analysis Ends")
            st.write("Want to try other model? click [here](#model)")
            st.write("Want to alter your question? click [here](#question)")

else:
    st.warning("Please select model at model section to proceed.")
