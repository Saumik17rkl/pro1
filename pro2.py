import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

# Title and description with improved layout
st.set_page_config(page_title="Iris Flower Prediction App", page_icon="🌸", layout="wide")
st.markdown(
    """
    <style>
    .main {background-color: #f0f8ff;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🌸 Iris Flower Prediction App")
st.write(
    """
    This interactive app lets you predict the type of Iris flower based on user-provided input parameters. 
    You can adjust the features in the sidebar to see how they influence the prediction.
    """
)

# Sidebar for user input
st.sidebar.header("🔧 Customize Parameters")
st.sidebar.markdown("Move the sliders to set the features of the Iris flower.")

def user_input_features():
    sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 0.2)
    data = {
        "Sepal Length (cm)": sepal_length,
        "Sepal Width (cm)": sepal_width,
        "Petal Length (cm)": petal_length,
        "Petal Width (cm)": petal_width,
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Input features
input_df = user_input_features()

# Display user input in a clean layout
st.subheader("Your Input Parameters")
st.dataframe(input_df.style.format(precision=2), use_container_width=True)

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Train a Random Forest Classifier
clf = RandomForestClassifier()
clf.fit(X, y)

# Make predictions
prediction = clf.predict(input_df)
prediction_proba = clf.predict_proba(input_df)

# Display predictions
col1, col2 = st.columns(2)

with col1:
    st.subheader("Prediction")
    st.success(f"The predicted class is **{iris.target_names[prediction][0]}** 🌼")

with col2:
    st.subheader("Prediction Probability")
    st.bar_chart(pd.DataFrame(prediction_proba, columns=iris.target_names))

# Additional Dataset Information
st.markdown("---")
st.subheader("About the Iris Dataset")
st.write(
    """
    The Iris dataset consists of 150 samples of three species of Iris flowers: 
    **Setosa**, **Versicolor**, and **Virginica**. Each sample includes four features:
    - **Sepal Length**
    - **Sepal Width**
    - **Petal Length**
    - **Petal Width**
    """
)

# Visualize the dataset with a scatter plot
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
for i, species in enumerate(iris.target_names):
    ax.scatter(
        X[y == i, 2],  # Petal length
        X[y == i, 3],  # Petal width
        label=species,
    )
ax.set_title("Petal Length vs. Petal Width")
ax.set_xlabel("Petal Length (cm)")
ax.set_ylabel("Petal Width (cm)")
ax.legend()
st.pyplot(fig)
