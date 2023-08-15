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


st.title("BREAST CANCER")
st.subheader("What is breast cancer?")

description = " \
                Breast cancer is a type of cancer that originates in the cells of the breast tissue. \
                It is the most common cancer diagnosed in women worldwide, although it can also occur in men, though it is much less common.  \
                Breast cancer usually starts in the milk ducts (ductal carcinoma) or the lobules (lobular carcinoma) of the breast. \
                "

type = "\
        There are two main types of lung cancer: Non-Invasive (or in situ) or Invasive. \
        Non-Invasive (or in situ)  if it stays localized to the area of the breast where it originates, without spreading through the surrounding breast tissue. \
        Then, for invasive when the neoplasm is able to migrate through the lymphatic system and blood and gradually compromise vital functions. \
        "

symptoms = "\
            The most common symptom of breast cancer is a new lump or mass (although most breast lumps are not cancer). \
            A painless, hard mass that has irregular edges is more likely to be cancer, but breast cancers can be also soft, round, tender, or even painful.\
            "
dataset_description = "\
            The datasets consist of mean radius, mean texture, mean perimeter, mean area, mean smoothness and diagnosis values for 1 = yes for diagnosis, and 0 = no diagnosis. \
            "

variable = "\
            The variable input will be used are all the datasets features except result of breast cancer,  \
            while the variable output will be result of breast cancer\
            \
            "

st.write(
    f'<p style="text-align: justify; font-size: 20px;">{description}</p>', unsafe_allow_html=True)
st.write(
    f'<p style="text-align: justify; font-size: 20px;">{type}</p>', unsafe_allow_html=True)

st.subheader("Symptoms")
st.write(
    f'<p style="text-align: justify; font-size: 20px;">{symptoms}</p>', unsafe_allow_html=True)

st.subheader("Breast Diagnosis")

col1, col2 = st.columns(2)
with col1:
    image = Image.open('images\Breast\Breast_cancer_1.jpg')
    newImage = image.resize((600, 600))
    st.image(newImage, caption='Outer')

with col2:
    image = Image.open('images\Breast\Breast_cancer_2.jpg')
    newImage = image.resize((600, 600))
    st.image(newImage, caption='Inner')

st.header("Analysis")
st.write(
    f'<p style="text-align: justify; font-size: 20px;">{"In this section, the web application is used to analysis whether the user has breast cancer or not."}</p>', unsafe_allow_html=True)

st.subheader("Dataset description")
st.write(
    f'<p style="text-align: justify; font-size: 20px;">{dataset_description}</p>', unsafe_allow_html=True)

st.subheader("Variable Input / Output")
st.write(
    f'<p style="text-align: justify; font-size: 20px;">{variable}</p>', unsafe_allow_html=True)

st.subheader("Model")
st.write(
    f'<p style="text-align: justify; font-size: 20px;">{"Pick models to use: KNN, Random Forest, SVM."}</p>', unsafe_allow_html=True)


dataset = pd.read_csv("dataset/Breast_cancer_data.csv")

selectModel = st.selectbox("Select model", options=[
                           "", "all", "KNN", "Random Forest", "SVM"])

data_input_training = dataset.drop(columns=["diagnosis"])
data_target_training = dataset["diagnosis"]

st.subheader("Training and Testing")
st.write(
    f'<p style="text-align: justify; font-size: 20px;">{"Training and Testing data will be divided using train test split."}</p>', unsafe_allow_html=True)

if (selectModel != ""):

    X_train, X_test, y_train, y_test = train_test_split(
        data_input_training, data_target_training, test_size=0.2)

    # if os.path.exists("joblib/tts_indices_breast_cancer.joblib"):
    #     X_train, X_test, y_train, y_test = joblib.load(
    #         'joblib/tts_indices_breast_cancer.joblib')
    # else:
    #     X_train, X_test, y_train, y_test = train_test_split(
    #         data_input_training, data_target_training, test_size=0.2)
    #     joblib.dump((X_train, X_test, y_train, y_test),
    #                 'joblib/tts_indices_breast_cancer.joblib')

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
            "What is the mean radius?", "What is the mean texture?", "What is the mean perimeter?", "What is the mean area?", "What is the mean smoothness?"
        ]

        st.subheader("Question")
        st.write(
            f'<p style="text-align: justify; font-size: 20px;">{"Question time. Please answer as honestly as possible."}</p>', unsafe_allow_html=True)

        for index, element in enumerate(question):

            answer = st.number_input(element)

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

                if (result == 1):
                    st.write(
                        f'<p style="text-align: justify; font-size: 20px;">{"Unfortunately, it appears that you have breast cancer."}</p>', unsafe_allow_html=True)
                else:
                    st.write(
                        f'<p style="text-align: justify; font-size: 20px;">{"Great! it does not seem that you have breast cancer"}</p>', unsafe_allow_html=True)

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
                if (result == 1):
                    st.write(
                        f'<p style="text-align: justify; font-size: 20px;">{"Unfortunately, it appears that you have breast cancer."}</p>', unsafe_allow_html=True)
                else:
                    st.write(
                        f'<p style="text-align: justify; font-size: 20px;">{"Great! it does not seem that you have breast cancer"}</p>', unsafe_allow_html=True)

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
                if (result == 1):
                    st.write(
                        f'<p style="text-align: justify; font-size: 20px;">{"Unfortunately, it appears that you have breast cancer."}</p>', unsafe_allow_html=True)
                else:
                    st.write(
                        f'<p style="text-align: justify; font-size: 20px;">{"Great! it does not seem that you have breast cancer"}</p>', unsafe_allow_html=True)

        if (value):
            st.subheader("Analysis Ends")
            st.write("Want to try other model? click [here](#model)")
            st.write("Want to alter your question? click [here](#question)")

else:
    st.warning("Please select model at model section to proceed.")
