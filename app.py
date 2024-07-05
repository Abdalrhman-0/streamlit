import streamlit as st
import pandas as pd
import numpy as np
import os
from pycaret.classification import setup as clf_setup, compare_models as compare_classification_models, pull as clf_pull
from pycaret.regression import setup as reg_setup, compare_models as compare_regression_models, pull as reg_pull
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
X = None
categorical_columns = None
task_type = None
best_model = None
columns_to_drop = None
def handle_missing_numerical_values(df, strategy):
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            if strategy == 'mean':
                df[col].fillna(df[col].mean(), inplace=True)
            elif strategy == 'median':
                df[col].fillna(df[col].median(), inplace=True)
            elif strategy == 'mode':
                 mode_values = df[col].mode()
                 if not mode_values.empty:
                    df[col].fillna(mode_values[0], inplace=True)
            elif strategy == 'drop rows':
                df.dropna(subset=[col], inplace=True)
    return df

def handle_missing_catigorical_values(df, strategy_catigorical):
     if strategy_catigorical == 'Mode':
        for column in df.select_dtypes(include=['object']).columns:
            df[column].fillna(df[column].mode()[0], inplace=True)
        st.write("DataFrame after filling missing values with mode")
        st.dataframe(df)
    
     elif strategy_catigorical == 'Additional Class':
        additional_class = st.text_input("Enter the additional class name for missing values", value="Missing")
        for column in df.select_dtypes(include=['object']).columns:
            df[column].fillna(additional_class, inplace=True)
        st.write(f"DataFrame after filling missing values with '{additional_class}'")
        st.dataframe(df)
     elif strategy_catigorical == 'drop rows':
        column = df.select_dtypes(include=['object']).columns
        df.dropna(subset= column, inplace=True)
        st.write("DataFrame after droping the rows that have a missing value")
        st.dataframe(df)

     return df

# Function to detect task type
def detect_task_type(y):
    if y.dtype == 'object' or y.nunique() < 20:
        return 'classification'
    else:
        return 'regression'

# Function to encode categorical features
def encode_categorical_features(df, columns, method):
    if method == 'Label Encoding':
        label_encoders = {}
        for column in columns:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            label_encoders[column] = le
        return df, label_encoders
    elif method == 'One-Hot Encoding':
        df = pd.get_dummies(df, columns=columns)
        return df, None

uploaded_file = st.file_uploader("Upload a file")
if uploaded_file is not None:
    file_extension = os.path.splitext(uploaded_file.name)[1]
if uploaded_file is None:
   st.error("uploud the data")
elif file_extension == '.csv':
    df = pd.read_csv(uploaded_file)
    st.subheader('Data Frame')
    st.dataframe(df)
    st.subheader("Summary Statistics")
    st.write(df.describe())
    st.write(df.shape)
    st.write(df.columns)
elif file_extension == '.excel':
    df = pd.read_excel(uploaded_file)
    st.subheader('Data Frame')
    st.dataframe(df)
    st.subheader("Summary Statistics")
    st.write(df.describe())
    st.write(df.shape)
    st.write(df.columns)
elif file_extension == '.json':
    df = pd.read_json(uploaded_file)
    st.subheader('Data Frame')
    st.json(df)
    st.subheader("Summary Statistics")
    st.write(df.describe())
    st.write(df.shape)
    st.write(df.columns)
else:
    st.error("Unsupported file format. Supported formats are: 'csv', 'excel', 'json'.")

columns_to_drop = st.multiselect("choose the columns you want to drop",options = df.columns)

def drop_columns():
    if columns_to_drop is not None:
        df.drop(columns = columns_to_drop , inplace=True)
        return df
    else:
        st.write("No columns selected for dropping")
drop_columns()

st.subheader("Select numerical Imputation Strategy")
strategy_numerical = st.selectbox("Choose imputation strategy:", ('mean', 'median', 'mode','drop rows'))
handle_missing_numerical_values(df, strategy_numerical)
st.dataframe(df)
st.subheader("Select catigorical Imputation Strategy")
strategy_catigorical = st.selectbox("Choose imputation strategy:", ("Additional Class", 'Mode','drop rows'))
handle_missing_catigorical_values(df, strategy_catigorical)
st.subheader("Catigorical data encoding")
target_column = st.selectbox("Select the target column (Y)", df.columns)
feature_columns = st.multiselect("Select feature columns (X)", df.columns.drop(target_column))

if target_column and feature_columns:
    X = df[feature_columns]
    Y = df[target_column]
    task_type = detect_task_type(Y)
    st.write("Detected task type:", task_type)
    processed_df = pd.concat([X, Y], axis=1)
else:
    st.write("Please select both feature columns (X) and a target column (Y).")


encoding_method = st.selectbox("Select encoding method for categorical columns", ['Label Encoding', 'One-Hot Encoding'])
if X is not None:
    categorical_columns = X.select_dtypes(include=['object']).columns
if categorical_columns is not None:
  if len(categorical_columns) > 0:   
    X, encoders = encode_categorical_features(X, categorical_columns, encoding_method)

if X is None:
    pass
else:
    st.write("Applying PyCaret...")

if task_type == 'classification':
    clf_setup(data=processed_df, target=target_column, session_id=123)
    best_model = compare_classification_models()
    model_results = clf_pull()
    accuracy = model_results.loc[model_results.index[0], 'Accuracy']
    st.write("Best model accuracy:", accuracy)
elif task_type == 'regression':
    reg_setup(data=processed_df, target=target_column, session_id=123)
    best_model = compare_regression_models()
    model_results = reg_pull()
    r2 = model_results.loc[model_results.index[0], 'R2']
    st.write("Best model R2:", r2)
else:
    st.write("Please select both feature columns (X) and a target column (Y).")

st.write("Best model:" ,best_model)
