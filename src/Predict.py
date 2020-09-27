########################################### IMPORT ALL THE REQUIRED LIBRARIES ###########################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime

from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error , r2_score, accuracy_score,confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, precision_recall_curve
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

def prepare_data1(train_file_name , test_file_name) :

    col = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land", "wrong_fragment","urgent","hot","num_failed_logins","logged_in", "num_compromised","root_shell","su_attempted","num_root","num_file_creations", "num_shells","num_access_files","num_outbound_cmds","is_hot_login", "is_guest_login","_count","srv_count","serror_rate", "srv_serror_rate", "rerror_rate","srv_rerror_rate","same_srv_rate", "diff_srv_rate", "srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate", "dst_host_diff_srv_rate","dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate", "dst_host_rerror_rate","dst_host_srv_rerror_rate","attack", "last_flag"]
    df = pd.read_csv(train_file_name, names = col)
    df_test = pd.read_csv(test_file_name, names = col)

    drop_cols_with_equal_min_max(df)   		# for test and train both
    drop_cols_with_equal_min_max(df_test)   # for test and train both

    # computing list of corelated columns which needs to be dropped
    df_corr = corelated_feature_matrix(df, threshold_value=0.7) # for train data
    drop_cols = compute_corelated_cols(df_corr)

    # Drop columns from Train and Test DataSet
    df.drop( columns= drop_cols , inplace = True)
    df_test.drop( columns= drop_cols , inplace = True)

    #encoding of categorical_cols
    categorical_cols = ['protocol_type', 'service', 'flag']
    encoding_categorical_cols(df, categorical_cols)
    encoding_categorical_cols(df_test, categorical_cols)

    # encoding of binary_cols
    binary_cols = ['land', 'logged_in', 'root_shell', 'is_hot_login', 'su_attempted']
    encoding_binary_cols(df, binary_cols)
    encoding_binary_cols(df_test, binary_cols)

    # encoding of target_cols
    target_cols = 'attack'
    encoding_target_cols(df_train , target_cols)
    encoding_target_cols(df_test , target_cols)

    target_cols = ['attack', 'attack_code'  , 'attack_type']
    feature_cols = df.drop(columns = target_cols).columns

    df_train_scaled = scaleData( df[feature_cols])
    df_test_scaled  = scaleData( df_test[feature_cols])

    imp_cols = compute_important_cols(df_train, min_importance_value = 0.02, plot=False)

    # dump  test and train data into csv file
    df_train_scaled[feature_cols].to_csv('train_data.csv', header = True , index=False , columns=dump_df.columns )
    df_test_scaled[feature_cols].to_csv('test_data.csv', header = True , index=False , columns=dump_df.columns )


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

def manual_test_input(df_train):
    """ Function to take Manual inputs for Network parameter """

    # make preparation for user input for network parameters
    inputDF = create_feature_inputs_sidebar(df_train)
    # st.text('manual_test_input .. start')
    # st.dataframe(inputDF.head())

    # prepare_test_as_train_data(df_train , inputDF)
    df_train['train'] = 1
    inputDF['train'] = 0
    df = pd.concat([df_train, inputDF], ignore_index=True)
    df, feature_col = lib.prepare_data(df, binary=True, category=True, scaling=True , target=False)
    df_train = df[ df['train'] == 1]
    inputDF =  df[ df['train'] == 0]
    df_train.drop(['train'], axis=1, inplace=True)
    inputDF.drop(['train'], axis=1, inplace=True)
    # inputDF.reset_index(drop=True,inplace=True)
    # st.text('manual_test_input .. end')

    return inputDF


def csv_test_input(df_train):
    """ Function to take File based inputs for Network parameter """

    uploaded_file = st.file_uploader(" Test Network Data", type="csv")
    inputDF = None
    if uploaded_file is not None:
        inputDF = pd.read_csv(uploaded_file)
        # st.text('csv_test_input .. start')
        # st.dataframe(inputDF.head())

        df_train['train'] = 1
        inputDF['train'] = 0
        df = pd.concat([df_train, inputDF], ignore_index=True)
        df, feature_col = lib.prepare_data(df, binary=True, category=True, scaling=True , target=False)
        df_train = df[ df['train'] == 1]
        inputDF =  df[ df['train'] == 0]
        df_train.drop(['train'], axis=1, inplace=True)
        inputDF.drop(['train'], axis=1, inplace=True)
        inputDF.reset_index(drop=True,inplace=True)
        # st.text('csv_test_input .. end')

    return inputDF

def predict_data(df_train, df_test):
    """ Function to load the already dumped model and predict the result """

    # st.dataframe(df_test)
    ## Load the model
    path = os.getcwd()

    st.write("Running Model")

    with open(path + '/../data/model.mdl', 'rb') as model_file :
        model = pickle.load(model_file)
        predict = lib.predict_data(model, df_test)

        return predict

def main() :
    st.title("""
     ** Predict Telecom Connection Status -  Attack or No-Attack ** !
    """)

    st.sidebar.header('Inputs')
    path = os.getcwd()

    # get the train data
    df_train = pd.read_csv(path + '/../data/train_data.csv')

    st.subheader("Manually input network parameter or from file")
    ret = st.radio(" Manually input network parameter or from file", ("Manual" , "File"))
    inputDF = None
    if ret is 'Manual':
        inputDF = manual_test_input(df_train)
    elif ret is 'File':
        inputDF = csv_test_input(df_train)


    # # get the list of all features
    # impdf = pd.read_csv(path + '/data/imp_features.csv')
    # impdf.columns = ['feature', 'importance']
    # impdf.set_index('feature', inplace=True)
    if inputDF is not None:

        predict = predict_data(df_train, inputDF, ret)
        st.subheader("Prediction")
        if ret is 'Manual':
            if predict:
                output = "Attack"
            else:
                output = "No-Attack"
            st.text(output)
        elif ret is 'File':
            inputDF['predicted_attack'] = predict
            st.dataframe(data = inputDF)

if __name__ == "__main__":
    # execute only if run as a script
    main()
