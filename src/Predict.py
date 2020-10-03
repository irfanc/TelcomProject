########################################### IMPORT ALL THE REQUIRED LIBRARIES ###########################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime

from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error , r2_score, accuracy_score,confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, precision_recall_curve, plot_confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.svm import SVC
# from xgboost import XGBClassifier
# from xgboost import plot_importance
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.neural_network import MLPClassifier
from IPython.display import Audio
import scipy
sound_file ="Neene Modalu.mp3"
import time
import streamlit as st
import Definitions as lib
import pickle
import os


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
            dict[col] = st.sidebar.slider(col , df[col].min() , df[col].max() , 1)
        else:
            dict[col] = st.sidebar.slider(col , df[col].min() , df[col].max(), 1.0)
    d = pd.DataFrame(dict, index=[0])

    return d

def prepare_data(df_train, df_test):
    """ Function to take prepare the input data for model prediction """

    df_train['train'] = 1
    df_test['train'] = 0
    df = pd.concat([df_train, df_test], ignore_index=True)
    df, feature_col = lib.prepare_data(df, binary=True, category=True, scaling=True , target=True)
    df_train = df[ df['train'] == 1]
    df_test = df[ df['train'] == 0]
    df.drop(['train'], axis=1, inplace=True)
    df_test.drop(['train'], axis=1, inplace=True)
    df_train.drop(['train'], axis=1, inplace=True)
    return df_train, df_test

def manual_test_input(df_train):
    """ Function to take Manual inputs for Network parameter """

    # make preparation for user input for network parameters
    inputDF = create_feature_inputs_sidebar(df_train)
    # st.text('manual_test_input .. start')
    # st.dataframe(inputDF.head())
    # df_test = prepare_data(df_train, inputDF)

    return inputDF


def csv_test_input(df_train):
    """ Function to take File based inputs for Network parameter """

    uploaded_file = st.sidebar.file_uploader("Telecom Network Testing Data", type="csv")
    inputDF = None
    if uploaded_file is not None:
        inputDF = pd.read_csv(uploaded_file)
        # st.text('csv_test_input .. start')
        # st.dataframe(inputDF.head())

        # df_test = prepare_data(df_train , inputDF)

    return inputDF

def predict_data(df_train, df_test, feature_col, target_col):
    """ Function to load the already dumped model and predict the result """

    # st.dataframe(df_test)
    path = os.getcwd()

    with open(path + '/../data/model.mdl', 'rb') as model_file :
        model = pickle.load(model_file)
        st.text(model)
        predict, confusion_matrix, class_rpt = lib.predict_data(model, df_test, feature_col, target_col)

        # compute model score score
        X = df_train[feature_col]
        Y = df_train[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=123)
        score = model.score(X_train, y_train)

        # plotting of confusion matrix
        # disp = plot_confusion_matrix(model, X, Y, cmap=plt.cm.Blues)
        # return predict, score, disp.confusion_matrix, class_rpt

        return predict, score, confusion_matrix, class_rpt


def main() :
    st.title("""
     ** Predict Telecom Connection Status -  Attack or No-Attack ** !
    """)

    st.sidebar.header('Inputs')
    path = os.getcwd()

    # get the train data
    df_train = pd.read_csv(path + '/../data/train_data.csv')
    FEATURE_COL = list(df_train.drop(columns='attack').columns)
    TARGET_COL = 'attack_code'

    # get the test data
    st.sidebar.subheader("Enter network parameter either manually or from file")
    ret = st.sidebar.radio(" ", ("Manual" , "File"))
    inputDF = None
    if ret is 'Manual':
        inputDF = manual_test_input(df_train)
    elif ret is 'File':
        inputDF = csv_test_input(df_train)

    if inputDF is not None:
        # prepare the data for prediction
        trainDF, testDF = prepare_data(df_train, inputDF)
        # st.dataframe(trainDF.describe().T )
        # st.dataframe(testDF.describe().T )

        # data prediction
        predict, score, conf_mat, classification_rpt = predict_data(trainDF, testDF, FEATURE_COL, TARGET_COL)
        st.subheader('Accuracy of the prediction is {}%'.format( round(100*score,2)))
        st.subheader ('Confusion Matrix')
        st.text (conf_mat)
        st.subheader("Classification Report")
        st.text (classification_rpt)

        st.subheader("Prediction")
        if ret is 'Manual':
            if predict:
                output = "There is attack on Network"
            else:
                output = "There is No-Attack in the Netwrok"
            st.text(output)
        elif ret is 'File':
            inputDF['predicted_attack'] = predict
            inputDF['predicted_attack'] = np.where(inputDF['predicted_attack'] == True, "Attack", "No-Attack")
            st.dataframe(data = inputDF)

        st.subheader("Test accuracy Score = ???")


if __name__ == "__main__":
    # execute only if run as a script
    main()
