import streamlit as st
import pandas as pd

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


st.title("Main Page")
st.header("Model")
st.write("Three models available: KNN, Random Forest, and SVM")

# selectDataset = st.selectbox("Select dataset", options= ["Male", "Female"])

# st.sidebar.success("Select a page above")

# datasetRead = ""

# if selectDataset == "Male":
#     datasetRead = pd.read_csv("dataset/xray_image_dataset_male.csv")
# elif selectDataset == "Female":
#     datasetRead = pd.read_csv("dataset/xray_image_dataset_female.csv")


# selectModel = st.selectbox("Select model", options=["KNN", "Random Forest", "SVM"])

# data_input_training =  datasetRead.drop(columns = ["No", "Race", "Gender", "DOB", "Exam Date", "Tanner", "Trunk HT (cm)"])
# data_target_training =  datasetRead['ChrAge']

# st.subheader("Training and testing data will be divided using train test split")
# X_train, X_test, y_train, y_test = train_test_split(data_input_training, data_target_training, test_size=0.2)

# st.write("data input training")
# X_train

# st.write("data target training")
# y_train

# st.write("data input testing")
# X_test

# st.write("data target testing")
# y_test

# if(selectDataset != "" and selectModel == "KNN"):
#     st.write("KNN")
#     knn = [1, 5, 10, 15]

#     for i in knn:
#         knn = KNeighborsRegressor(n_neighbors=i)
#         knn.fit(X_train, y_train)

#         st.write("prediction knn:", i)
#         prediction = knn.predict(X_test)
#         prediction

#         st.write("mean squared error: ", i)
#         knnEvaluation = mean_squared_error(prediction, y_test)
#         knnEvaluation

# elif(selectDataset != "" and selectModel == "Random Forest"):
#     st.write("random forest")
   
#     estimators = [100, 500, 1000]

#     for i in estimators:
#         st.write("estimator", i)
#         st.write("prediction")
        
#         rf = RandomForestRegressor(n_estimators=i, random_state=0)
#         rf.fit(X_train, y_train)

#         prediction = rf.predict(X_test)
#         prediction

#         st.write("mean squared error")
#         randomForest = mean_squared_error(prediction, y_test)
#         randomForest


# elif(selectDataset != "" and selectModel == "SVM"):

#     kernel = ["linear", "rbf", "poly", "sigmoid"]
#     svm_models = []

#     st.write("SVM")

#     for i in kernel:
#         svm_model = SVR(kernel=i)
#         svm_model.fit(X_train, y_train)

#         st.write("SVM", i, "Prediction")
#         prediction = svm_model.predict(X_test)

#         st.write("mean squared error: for kernel", i)
#         svm = mean_squared_error(prediction, y_test)
#         svm








