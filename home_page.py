import streamlit as st
import pandas as pd
import streamlit.components.v1 as components

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR



st.title("Home")

st.subheader("Model")
st.write(f'<p style="text-align: justify; font-size: 20px; padding-bottom: 20px;">{"Three models available: KNN, Random Forest, and SVM"}</p>', unsafe_allow_html=True)

st.subheader("Results")

data = [
    {
        'title': "Lung Cancer",
        'description': [
            {
                'model': "KNN",
                'option': ["1", "5", "10", "15"],
                'results':   ["0.8870967741935484", "0.8709677419354839", "0.8225806451612904", "0.8548387096774194"],
            },
             {
                'model': "RF",
                'option': ["100", "500", "1000"], 
                'results': ["0.9032258064516129", "0.9032258064516129", "0.8709677419354839"], 
            },
             {
                'model': "SVM",
                'option': ["linear", "rbf", "poly", "sigmoid"], 
                'results': ["0.9193548387096774", "0.8548387096774194", "0.8548387096774194", "0.8548387096774194"], 
            },
        ]
    },
     {
        'title': "Breast Cancer",
        'description': [
            {
                'model': "KNN",
                'option': ["1", "5", "10", "15"],
                'results':   ["1", "5", "10", "15"],
            },
             {
                'model': "RF",
                'option': ["100", "500", "1000"], 
                'results': ["100", "500", "1000"], 
            },
             {
                'model': "SVM",
                'option': ["linear", "rbf", "poly", "sigmoid"], 
                'results': ["linear", "rbf", "poly", "sigmoid"], 
            },
        ]
    },
    {
        'title': "Kidney Cancer",
        'description': [
            {
                'model': "KNN",
                'option': ["1", "5", "10", "15"],
                'results':   ["1", "5", "10", "15"],
            },
             {
                'model': "RF",
                'option': ["100", "500", "1000"], 
                'results': ["100", "500", "1000"], 
            },
             {
                'model': "SVM",
                'option': ["linear", "rbf", "poly", "sigmoid"], 
                'results': ["linear", "rbf", "poly", "sigmoid"], 
            },
        ]
    },
    {
        'title': "Prostate Cancer",
        'description': [
            {
                'model': "KNN",
                'option': ["1", "5", "10", "15"],
                'results':   ["1", "5", "10", "15"],
            },
             {
                'model': "RF",
                'option': ["100", "500", "1000"], 
                'results': ["100", "500", "1000"], 
            },
             {
                'model': "SVM",
                'option': ["linear", "rbf", "poly", "sigmoid"], 
                'results': ["linear", "rbf", "poly", "sigmoid"], 
            },
        ]
    },
    {
        'title': "Skin Cancer",
        'description': [
            {
                'model': "KNN",
                'option': ["1", "5", "10", "15"],
                'results':   ["1", "5", "10", "15"],
            },
             {
                'model': "RF",
                'option': ["100", "500", "1000"], 
                'results': ["100", "500", "1000"], 
            },
             {
                'model': "SVM",
                'option': ["linear", "rbf", "poly", "sigmoid"], 
                'results': ["linear", "rbf", "poly", "sigmoid"], 
            },
        ]
    },
]

for element in data:

    title = element['title']
    st.write(f'<p style="text-align: justify; font-size: 30px; font-weight: bold;">{title}</p>', unsafe_allow_html=True)
   
    for i, k in enumerate (element['description']):
        model = k['model']
        st.write(f'<p style="text-align: justify; font-size: 20px; font-weight: bold; font-style: italic;">{model}</p>', unsafe_allow_html=True)
        
        if(i==0):
            num_columns = 5
            columns = [st.columns(num_columns, gap="large") for _ in range(num_columns)]
            columns[0][0].write("neighbors")
            columns[1][0].write("mse")
            # columns_result = [st.columns(num_columns) for _ in range(num_columns)]
            # columns_result[0][0].write("results")
            # c1, c2, c3, c4 = st.columns(4)
        elif i == 1:
            num_columns = 4
            columns = [st.columns(num_columns, gap="large") for _ in range(num_columns)]
            columns[0][0].write("estimators")
            columns[1][0].write("mse")
            # columns_result = [st.columns(num_columns) for _ in range(num_columns)]
            # columns_result[0][0].write("results")
        elif i == 2: 
            num_columns = 5
            columns = [st.columns(num_columns, gap="large") for _ in range(num_columns)]
            columns[0][0].write("kernels")
            columns[1][0].write("mse")
            # columns_result = [st.columns(num_columns) for _ in range(num_columns)]
            # columns_result[0][0].write("results")
        # st.write("option")
        for l, option in enumerate (k['option']):
            if(i==0):
                # name = "c"+ str(l+1)
                with st.container():
                    columns[0][l+1].write(option)
            if(i==1):
                # name = "c"+ str(l+1)
                with st.container():
                    columns[0][l+1].write(option)
            if(i==2):
                # name = "c"+ str(l+1)
                with st.container():
                    columns[0][l+1].write(option)
                
                # st.write(name)
                # c1.write(option)
                # c2.write(option)
                # c3.write(option)
                # c4.write(option)
            # st.write()
        # for l in range(num_columns):
        #     st.write(l)
        # st.write("results")
        for o, results in enumerate (k['results']):
            if(i==0):
                # name = "c"+ str(l+1)
                with st.container():
                    columns[1][o+1].write(results)
            if(i==1):
                # name = "c"+ str(l+1)
                with st.container():
                    columns[1][o+1].write(results)
            if(i==2):
                # name = "c"+ str(l+1)
                with st.container():
                    columns[1][o+1].write(results)
    # components.html(
    # """
    # <hr>
    # """,
    # height=10
    # )






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








