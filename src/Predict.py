########################################### IMPORT ALL THE REQUIRED LIBRARIES ###########################################
import numpy as np
import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt
# import datetime

# from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
# from sklearn.metrics import mean_squared_error , r2_score, accuracy_score,confusion_matrix, classification_report
# from sklearn.metrics import roc_curve, roc_auc_score, f1_score, precision_recall_curve
# from sklearn.metrics import plot_confusion_matrix, ConfusionMatrixDisplay
# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
# from sklearn import model_selection
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.naive_bayes import GaussianNB
# from sklearn.model_selection import KFold
# from sklearn.svm import SVC
# from xgboost import XGBClassifier
# from xgboost import plot_importance
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# from sklearn.neural_network import MLPClassifier
# from IPython.display import Audio
# import scipy
# sound_file ="Neene Modalu.mp3"
# import time
import streamlit as st
import Definitions as lib
import pickle
import os, io


FEATURE_COL = None
TARGET_COL = None

def create_feature_inputs_sidebar(df):
    """ Creates a layout so that used can input the Network parameters
    This function returns the a Data Frame of the User input Networ parameter """

    # lib.drop_cols_with_equal_min_max(df)  # for test and train both

    # encoding of binary_cols
    binary_cols = ['land', 'logged_in', 'root_shell', 'is_hot_login', 'su_attempted']
    df[binary_cols] = df[binary_cols].astype('bool')
    # encoding_binary_cols(DF, binary_cols)
    dict = {}
    for col in df.columns :
        if df[col].dtype.name == 'object':
            dict[col] = st.sidebar.selectbox(col, df[col].unique())
        elif df[col].dtype.name == 'bool':
            dict[col] = st.sidebar.checkbox(col)
        elif df[col].dtype.name == 'int64':
            dict[col] = st.sidebar.number_input(col, df[col].min() , df[col].max() , 1)
        else:
            dict[col] = st.sidebar.number_input(col, df[col].min() , df[col].max(), 1.0)
    d = pd.DataFrame(dict, index=[0])

    return d

def prepare_data(df_train, df_test, feature_col, target_col, binary=True, category=True, target=True):
    """ Function to take prepare the input data for model prediction """
    train, test = lib.prepare_test_train_data(df_train, df_test, binary=binary, category = category, scaling = False, target = target)
    df_train_scaled   = lib.scaleData(train[feature_col])
    df_test_scaled    = lib.scaleData(test[feature_col])

    return df_train_scaled, train[target_col], df_test_scaled, test[target_col]

def manual_test_input(df_train):
    """ Function to take Manual inputs for Network parameter """

    # make preparation for user input for network parameters
    inputDF = create_feature_inputs_sidebar(df_train)

    return inputDF

def csv_test_input(df_train):
    """ Function to take File based inputs for Network parameter """
    uploaded_file = st.sidebar.file_uploader("Telecom Network Test-Data", type="csv")
    if uploaded_file is None:
        return None

    # st.text ("File is closed ? --> {} ".format(uploaded_file.closed))
    uploaded_file.seek(0)
    inputDF = pd.read_csv(uploaded_file)
    # st.text ("File is closed ? --> {} ".format(uploaded_file.closed))
    return inputDF

@st.cache()
def predict_data(Xtrain, Ytrain, Xtest, Ytest):
    """ Function to load the already dumped model and predict the result """

    path = os.getcwd()
    with open(path + '/../data/model.mdl', 'rb') as model_file :
        model = pickle.load(model_file)
        # st.text(model)
        predict, confusion_matrix, class_rpt = lib.predict_data(model, Xtest, Ytest)

        # compute model score score
        x_train, x_test, y_train, y_test = train_test_split(Xtrain, Ytrain, test_size=0.3, random_state=123)
        score = model.score(Xtest, Ytest)

        return predict, score, confusion_matrix, class_rpt

MANUAL_INPUT = "Manual Input"
FILE_INPUT = "Input Test File"

@st.cache(allow_output_mutation=True)
def get_train_data():
    # get the train data
    path = os.getcwd()
    df_train = pd.read_csv(path + '/../data/train_data.csv')

    return df_train

def get_input_data(inputType, df_train):
    if inputType is MANUAL_INPUT:
        # st.text('Preparing Manual Input')
        inputDF = manual_test_input(df_train)
    elif inputType is FILE_INPUT:
        # st.text('Preparing File Input')
        inputDF = csv_test_input(df_train)

    return inputDF


def show_prediction_details (conf_mat, classification_rpt) :
    st.subheader('Confusion Matrix')
    st.text (conf_mat)
    st.subheader("Classification Report")
    st.text (classification_rpt)
    st.text ("")
    st.text ("")

@st.cache()
def display_test_data(inputDF, predict):
    inputDF['predicted_attack'] = predict
    inputDF['predicted_attack'] = np.where(inputDF['predicted_attack'] == True, "Attack", "No-Attack")
    return inputDF

def main():
    st.title("""
     ** Predict Telecom Connection Status -  Attack or No-Attack ** !
    """)

    st.sidebar.header('Inputs')

    # get the train data
    df_train = get_train_data()
    FEATURE_COL = list(df_train.drop(columns=['attack']).columns)
    TARGET_COL = 'attack_code'

    # get the test data
    st.sidebar.subheader("Enter network parameter either manually or from file")
    ret = st.sidebar.radio(" Input Type ", (MANUAL_INPUT , FILE_INPUT))
    inputDF = get_input_data(ret , df_train)

    if inputDF is not None:
        # prepare the data for prediction
        Xtrain, Ytrain, Xtest, Ytest = prepare_data(df_train.copy(), inputDF, feature_col=FEATURE_COL, target_col=TARGET_COL)

        # data prediction
        predict, score, conf_mat, classification_rpt = predict_data(Xtrain, Ytrain, Xtest, Ytest)

        st.subheader("Prediction")
        if ret is MANUAL_INPUT:
            st.text(predict)
            if predict is True:
                output = "There is an Attack "
            else:
                output = "There is No-Attack"
            st.text(output)
        elif ret is FILE_INPUT:
            inputDF = display_test_data(inputDF, predict)
            st.dataframe(inputDF)

        if ret is FILE_INPUT and st.checkbox(" Prediction Details ") :
            show_prediction_details(conf_mat, classification_rpt)

if __name__ == "__main__":
    # execute only if run as a script
    main()
