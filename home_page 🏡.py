import streamlit as st
import streamlit.components.v1 as components


st.title("Home")

description = "The web application is used to determine cancers such as breast, kidney, lung, prostate, and skin.\
                For breast cancer, the web application is going to determine whether the user has lung cancer or not. \
                Next, for kidney the web application is going to determine whether the user has lung cancer or not. \
                After that, lung cancer, the web application is going to determine whether the user has lung cancer or not. \
                Other than that, for prostate cancer, the web application is going to determine whether the user has lung cancer or not.\
                Lastly, for skin cancer, the web application is going to determine whether the user has lung cancer or not.\
                These predictions are possible using solution which is by questioning the users of their condition or simply by  \
                input physical numerical value according to the requirements and using machine learning to determine the possible answer.  \
                "

st.subheader("Description of the system")
st.write(
    f'<p style="text-align: justify; font-size: 20px; padding-bottom: 20px;">{description}</p>', unsafe_allow_html=True)

to_use = "\n 1. From the appbar at left choose the cancer to try \n 2. Pick the model to try \n 3. Answer the question \n 4. Click the start analysis button"

st.subheader("Model & How to use")
st.write(
    f'<p style="text-align: justify; font-size: 20px;">{"Three models available: KNN, Random Forest, and SVM"}</p>', unsafe_allow_html=True)

st.write(
    f'<p style="text-align: justify; font-size: 20px;">{"How to use the web application: "}</p>', unsafe_allow_html=True)
st.write(to_use)

st.write(
    f'<p style="text-align: justify; font-size: 20px;">{""}</p>', unsafe_allow_html=True)

st.subheader("Result Conclusion")

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
                'results':   ["0.8596491228070176", "0.8859649122807017", "0.8947368421052632", "0.9035087719298246"],
            },
            {
                'model': "RF",
                'option': ["100", "500", "1000"],
                'results': ["0.9473684210526315", "0.9473684210526315", "0.9473684210526315"],
            },
            {
                'model': "SVM",
                'option': ["linear", "rbf", "poly", "sigmoid"],
                'results': ["0.9385964912280702", "0.8947368421052632", "0.8859649122807017", "0.42105263157894735"],
            },
        ]
    },
    {
        'title': "Kidney Cancer",
        'description': [
            {
                'model': "KNN",
                'option': ["1", "5", "10", "15"],
                'results':   ["0.775", "0.7", "0.725", "0.7625"],
            },
            {
                'model': "RF",
                'option': ["100", "500", "1000"],
                'results': ["0.9875", "0.9875", "0.9875"],
            },
            {
                'model': "SVM",
                'option': ["linear", "rbf", "poly", "sigmoid"],
                'results': ["0.9875", "0.675", "0.675", "0.6"],
            },
        ]
    },
    {
        'title': "Prostate Cancer",
        'description': [
            {
                'model': "KNN",
                'option': ["1", "5", "10", "15"],
                'results':   ["0.85", "0.85", "0.8", "0.8"],
            },
            {
                'model': "RF",
                'option': ["100", "500", "1000"],
                'results': ["0.8", "0.8", "0.8"],
            },
            {
                'model': "SVM",
                'option': ["linear", "rbf", "poly", "sigmoid"],
                'results': ["0.85", "0.8", "0.75", "0.4"],
            },
        ]
    },
    {
        'title': "Skin Cancer",
        'description': [
            {
                'model': "KNN",
                'option': ["1", "5", "10", "15"],
                'results':   ["0.9588792582140697", "0.9689578713968958", "0.9667405764966741", "0.9639185648054828"],
            },
            {
                'model': "RF",
                'option': ["100", "500", "1000"],
                'results': ["0.970167304978835", "0.9703688772424914", "0.9703688772424914"],
            },
            {
                'model': "SVM",
                'option': ["linear", "rbf", "poly", "sigmoid"],
                'results': ["0.9439629107034871", "0.928038701874622", "0.9330780084660351", "0.8713968957871396"],
            },
        ]
    },
]

best_algo = ""

for element in data:

    title = element['title']
    st.write(
        f'<p style="text-align: justify; font-size: 25px; font-weight: bold;">{title}</p>', unsafe_allow_html=True)

    for i, k in enumerate(element['description']):
        model = k['model']
        st.write(
            f'<p style="text-align: justify; font-size: 20px; font-weight: bold; font-style: italic;">{model}</p>', unsafe_allow_html=True)

        option = ""

        if model == "KNN":
            num_columns = 5
            option = "neighbors"
        elif model == "RF":
            num_columns = 4
            option = "estimators"
        elif model == "SVM":
            num_columns = 5
            option = "kernels"

        columns = [st.columns(num_columns, gap="large")
                   for _ in range(num_columns)]

        columns[0][0].write(option)
        columns[1][0].write("accuracy")

        for l, option in enumerate(k['option']):
            with st.container():
                columns[0][l+1].write(option)

        for o, results in enumerate(k['results']):
            with st.container():
                columns[1][o+1].write(results)

        if (title == "Lung Cancer"):
            best_algo = "SVM with kernel linear"
        elif (title == "Breast Cancer"):
            best_algo = "Random Forest with estimators 100, 500, 1000"
        elif (title == "Kidney Cancer"):
            best_algo = "Random Forest with estimators 100, 500, 1000 and SVM with linear kernel"
        elif (title == "Prostate Cancer"):
            best_algo = "KNN with 1, 5 neigbors and SVM with kernel linear"
        elif (title == "Skin Cancer"):
            best_algo = "Random Forest with estimators 500, 1000"

    st.subheader("best algorithm:")
    st.write(
        f'<p style="text-align: justify; font-size: 19px; font-weight: bold; font-style: italic;">{best_algo}</p>', unsafe_allow_html=True)

    st.write(
        f'<p style="text-align: justify; font-size: 18px; font-weight: bold; font-style: italic;">{"closest accuracy to 1 means the accuracy is far precise"}</p>', unsafe_allow_html=True)

    st.write(
        f'<p style="text-align: justify; font-size: 20px; padding-bottom: 30px;">{""}</p>', unsafe_allow_html=True)

    best_algo = ""
    components.html(
        """
    <hr>
    """,
        height=10
    )
