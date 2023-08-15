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

st.title("SKIN CANCER")
st.subheader("What is skin cancer?")

description = " \
                Skin cancer is a type of cancer that originates in the skin cells. It occurs when certain skin cells undergo uncontrolled growth and form tumors.   \
               The most common cause of skin cancer is exposure to ultraviolet (UV) radiation from the sun or tanning beds.  \
                However, genetic factors and certain environmental exposures can also contribute to its development.\
                "

type = "\
        There is three types of skin cancer, which is Basal Cell Carcinoma (BCC),Squamous Cell Carcinoma (SCC) and Melanoma. \
        Basal Cell Carcinoma (BCC) is the most common type of skin cancer, it usually appears on areas of the skin that are frequently exposed to the sun.\
        "

symptoms = "\
           Skin cancer symptoms include changes in the appearance of moles or spots, such as asymmetry, \
           irregular borders, color changes, or growth in size. Watch out for new growths or lumps on the skin that don't heal, \
            and may appear as shiny, pearly bumps or red, scaly patches.\
            "
dataset_description = "\
            The dataset consist of cld(0-100), dtr(0-50), frs(0-50), pet(0-10), pre(0-300), tmn(-30-30), \
             tmp(-30-30), tmx(-10-30), vap(0-20), wet(0-20), elevation(1-300), dominant land cover(0-15), lumpy (0-1), x coordinate, \
            y coordinate, region, country, reporting date, X5_Ct_2010_Da, X5_Bf_2010_Da.\
            "

variable = "\
            The variable input will be used are all the datasets features except the x coordinate, y coordinate, region, country, reporting date, X5_Ct_2010_Da, X5_Bf_2010_Da and lumpy  \
            while the lumpy will be result of potential place skin cancer exposure\
            \
            "
st.write(
    f'<p style="text-align: justify; font-size: 20px;">{description}</p>', unsafe_allow_html=True)
st.write(
    f'<p style="text-align: justify; font-size: 20px;">{type}</p>', unsafe_allow_html=True)

st.subheader("Symptoms")
st.write(
    f'<p style="text-align: justify; font-size: 20px;">{symptoms}</p>', unsafe_allow_html=True)

st.subheader("Skin Diagnosis")

col1, col2 = st.columns(2)
with col1:
    image = Image.open('images\skin\skincancer1.jpg')
    newImage = image.resize((600, 600))
    st.image(newImage, caption='internal')

with col2:
    image = Image.open('images\skin\skincancer2.jpg')
    newImage = image.resize((600, 600))
    st.image(newImage, caption='external')

st.header("Analysis")
st.write(
    f'<p style="text-align: justify; font-size: 20px;">{"In this section, the web application is used to analysis whether the living place has potential of skin cancer exposure."}</p>', unsafe_allow_html=True)

st.subheader("Dataset description")
st.write(
    f'<p style="text-align: justify; font-size: 20px;">{dataset_description}</p>', unsafe_allow_html=True)

st.subheader("Variable Input / Output")
st.write(
    f'<p style="text-align: justify; font-size: 20px;">{variable}</p>', unsafe_allow_html=True)

st.subheader("Model")
st.write(
    f'<p style="text-align: justify; font-size: 20px;">{"Pick models to use: KNN, Random Forest, SVM."}</p>', unsafe_allow_html=True)


dataset = pd.read_csv("dataset/skin_cancer.csv")

selectModel = st.selectbox("Select model", options=[
                           "", "all", "KNN", "Random Forest", "SVM"])

data_input_training = dataset.drop(columns=["x", "y",
                                   "region", "country", "reportingDate", "X5_Ct_2010_Da", "X5_Bf_2010_Da", "lumpy"])
data_target_training = dataset["lumpy"]

st.subheader("Training and Testing")
st.write(
    f'<p style="text-align: justify; font-size: 20px;">{"Training and Testing data will be divided using train test split."}</p>', unsafe_allow_html=True)

if (selectModel != ""):

    X_train, X_test, y_train, y_test = train_test_split(
        data_input_training, data_target_training, test_size=0.2)

    # if os.path.exists("joblib/tts_indices_skin_cancer.joblib"):
    #     X_train, X_test, y_train, y_test = joblib.load(
    #         'joblib/tts_indices_skin_cancer.joblib')
    # else:
    #     X_train, X_test, y_train, y_test = train_test_split(
    #         data_input_training, data_target_training, test_size=0.2)
    #     joblib.dump((X_train, X_test, y_train, y_test),
    #                 'joblib/tts_indices_skin_cancer.joblib')

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
            "what is your cloud cover?", "what is the  diurnal temperature range?", "what is your frequency?",
            "what is your potential evapotranspiration?", "your precipitation?", "your minimum temperature?", "your current air tempreture?", "your max tempreture?",
            "what is your water vapor pressure?", "your wetness index?", "what is your elevation?", "what is your dominant land cover?",
        ]

        st.subheader("Question")
        st.write(
            f'<p style="text-align: justify; font-size: 20px;">{"Question time. Please answer as honestly as possible."}</p>', unsafe_allow_html=True)

        for index, element in enumerate(question):

            if (element == "what is your elevation?"):
                answer = st.slider(element, 0, 300)
            elif (element == "what is your dominant land cover?"):
                answer = st.slider(element, 0, 50)
            else:
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
                        f'<p style="text-align: justify; font-size: 20px;">{"It appears that your location could have potential of skin cancer exposure."}</p>', unsafe_allow_html=True)
                else:
                    st.write(
                        f'<p style="text-align: justify; font-size: 20px;">{"It appears that your location is safe from skin cancer exposure."}</p>', unsafe_allow_html=True)

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
                        f'<p style="text-align: justify; font-size: 20px;">{"It appears that your location could have potential of skin cancer exposure."}</p>', unsafe_allow_html=True)
                else:
                    st.write(
                        f'<p style="text-align: justify; font-size: 20px;">{"It appears that your location is safe from skin cancer exposure."}</p>', unsafe_allow_html=True)

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
                        f'<p style="text-align: justify; font-size: 20px;">{"It appears that your location could have potential of skin cancer exposure."}</p>', unsafe_allow_html=True)
                else:
                    st.write(
                        f'<p style="text-align: justify; font-size: 20px;">{"It appears that your location is safe from skin cancer exposure."}</p>', unsafe_allow_html=True)

        if (value):
            st.subheader("Analysis Ends")
            st.write("Want to try other model? click [here](#model)")
            st.write("Want to alter your question? click [here](#question)")

else:
    st.warning("Please select model at model section to proceed.")
