########################################### IMPORT ALL THE REQUIRED LIBRARIES ###########################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
# get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
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

sound_file = "Neene Modalu.mp3"
import time
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process, \
    metrics
import pickle


# In[3]:


# from google.colab import drive
# import os
# drive.mount('/gdrive')
# # %cd /gdrive
# print( os.getcwd())


# In[4]:

def load_test_train(train_file_name, test_file_name):
    col = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
           "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
           "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_hot_login",
           "is_guest_login", "_count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
           "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
           "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
           "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
           "dst_host_srv_rerror_rate", "attack", "last_flag"]
    df = pd.read_csv(train_file_name, names=col)
    df_test = pd.read_csv(test_file_name, names=col)
    return df, df_test


def plt_missing_value_percentage(DF, title):
    plt.figure(figsize=(15, 5))
    print(DF.isnull().sum() * 100 / len(DF))
    ax = sns.heatmap(data=DF.isna(), cbar=False, cmap='BuPu_r', yticklabels=False)
    ax.set_title(title)
    plt.show()


def autoLabel(ax, fontsize=16):
    for rect in ax.patches:
        height = 0
        width = 0
        if (rect.get_height()):
            height = rect.get_height()
            width = rect.get_width()
        ax.text(rect.get_x() + width / 2., 1.005 * height, str(height), ha='center', va='bottom', fontsize=fontsize)

def check_attack_class_density(df):
    fig, ax = plt.subplots(2, 1, figsize=(20, 20))
    fig.suptitle('Target Class distribution', fontsize=50)  # Add the text/suptitle to figure

    attack_df = df["attack"].value_counts()[1:]
    l = []
    for x in attack_df:
        if (x < 10):
            l.append(0.5)
        elif (x < 10):
            l.append(0.4)
        elif (x < 50):
            l.append(0.3)
        elif (x < 1000):
            l.append(0.1)
        else:
            l.append(0)

    explode = tuple(l)

    # frrequency graph of 'attack'
    sns.countplot('attack', data=df, order=attack_df.index, ax=ax[0])
    ax[0].set_ylabel('# of Attacks', fontsize=18)
    ax[0].set_xlabel('Type of Attacks', fontsize=18)
    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=90, fontsize=18)
    ax[0].set_yticklabels(ax[0].get_yticklabels(), fontsize=16)

    autoLabel(ax[0])

    # pie chart of 'attack'
    attack_df.plot.pie(ax=ax[1], autopct='%1.2f%%', explode=explode, shadow=True, startangle=45,
                       textprops={'fontsize': 18})
    ax[1].set_aspect(aspect='auto')
    plt.tight_layout()
    plt.show()


def drop_cols_with_equal_min_max(DF):
    """From Panda profiling, we found that there is a column with all the values are same.
    Below code is to find if max and min of a column is same then it contains all same values
    And we have decided to drop such columns, becasue it will cause no impact on the target variable
    becasue there is no change in value of this column for different values of target variable"""
    df_t = DF.describe().T
    drop_cols = df_t[df_t['min'] == df_t['max']].index.to_list()
    print('No. of colums with all value to be same - {} , {} '.format(len(drop_cols), str(drop_cols)))
    if (len(drop_cols) is not 0):
        DF.drop(columns=drop_cols, inplace=True)


def plot_correlated_cols(DF):
    """Checking the co-relation between the Xs
    """
    fig, ax = plt.subplots(1, 1, figsize=(30, 30))
    fig.suptitle('Correlation Matrix', fontsize=50)  # Add the text/suptitle to figure
    sns.heatmap(DF, annot=True, cmap='YlGnBu', fmt='.2g', square=True)
    plt.show()


def corelated_feature_matrix(DF, threshold_value=0.7):
    """returns correlated Matrix, with 1 representing corelation and 0- indicated no corelation,
    criteria of corelation is threshold_value of corelation, is corelation value > threshold_value then cols are said to be corelated """
    corr = round(DF.corr(), 3)
    corr1 = np.where((abs(corr) > threshold_value), 1, 0)
    df_corr = pd.DataFrame(corr1, columns=corr.columns, index=corr.columns)
    return df_corr

# returns list of corelated columns which we can drop
# input is correlated matrix
def compute_corelated_cols(DF):
    dict = {}
    # iterate over all the columns to find out others which are correlated > 0.7,  as per corrlation matrix
    for c in DF.columns:
        l = DF.index[DF[c] == 1].tolist()
        l.remove(c)  #  remove slef column from the list
        #     print(l)
        if len(l) :
            dict[c] = set(l)

    drop_set = set()
    for e in dict:
    #     print(e , dict[e] , drop_set)
      if e not in drop_set :
            drop_set = drop_set.union(dict[e])

    print("No. of corelated columns needs to be dropped  are  {}\n{}".format(len(drop_set), str(drop_set)))
    return drop_set

def encoding_binary_cols(DF, binary_cols):
    '''encoding binary columns'''
    DF[binary_cols] = DF[binary_cols].astype('int8')


def encoding_categorical_cols(DF, categorical_cols):
    """encoding categorical columns"""
    from sklearn.preprocessing import LabelEncoder

    lb_make = LabelEncoder()
    for c in categorical_cols:
        DF[c] = lb_make.fit_transform(DF[c])

    # print(DF[categorical_cols].info())


def encoding_target_cols(DF, target_cols=str):
    """ encode 'normal' attack as 0 , i.e Fasle  and othere attacks as True attack"""
    # Below code converts a multiclass categorical column into a binary col, where replace 'normal' with Flase and rest with True
    DF[target_cols + '_code'] =  np.where(DF[target_cols].str.contains("normal"), False, True)
    # DF[target_cols] = DF[target_cols].astype('category')


def plot_important_cols(DF, col, imp, min_importance_value=0.02):
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(' Feature Importance Chart', fontsize=50)  # Add the text/suptitle to figure

    ax = sns.barplot(col, imp)
    ax.set_xlabel(" Columns ", fontsize=18)
    ax.set_ylabel(" Columns relative importance", fontsize=20)
    ax.set_xticklabels(col, rotation=90, fontsize=20)

    ax.grid(b=True, which='both', color='b', linestyle='-')
    ax.grid(b=True, which='minor', color='r', linestyle=':')
    ax.minorticks_on()
    ax.set_axisbelow(True)

    ax.axhline(min_importance_value, color='red')
    ax.text(-3, min_importance_value + 0.01, "Minimum threshold importance value\n for feature selection ", color='red',
            fontsize=20)
    plt.show()


# feature selection by Random Forest
def compute_important_cols(DF, feature_cols, min_importance_value=0.02, plot=False):
    X = DF[feature_cols]
    Y = DF['attack_code']
    model = RandomForestClassifier(random_state=123, max_depth=20)
    model.fit(X, Y)
    imp = model.feature_importances_

    imp_features = []
    for i, f in zip(imp, feature_cols):
        if i > min_importance_value:
            imp_features.append(f)
    print("Out of {} features based on relative importance = {} selecting {} features".format(len(feature_cols),
                                                                                              min_importance_value,
                                                                                              len(imp_features)))
    if (plot):
        plot_important_cols(DF, X.columns, imp)
    return imp_features


from sklearn import preprocessing

def scaleData(DF):
    scl = preprocessing.MinMaxScaler(feature_range=(0, 1))
    arr_scld = scl.fit_transform(DF)
    d_scld = pd.DataFrame(arr_scld)
    d_scld.columns = DF.columns
    #     print(d_scld.describe())
    return d_scld

def prepare_data(df, binary=True, category=True, target=False, scaling=False):
    DF = df.copy()

    if category:
        # encoding of categorical_cols
        categorical_cols = ['protocol_type', 'service', 'flag']
        encoding_categorical_cols(DF, categorical_cols)

    if binary:
        # encoding of binary_cols
        binary_cols = ['land', 'logged_in', 'root_shell', 'is_hot_login', 'su_attempted']
        encoding_binary_cols(DF, binary_cols)

    col = ['attack']

    if target:
        # encoding of target_cols
        target_cols = 'attack'
        print( DF.attack.dtype.name )
        encoding_target_cols(DF, target_cols)
        col = ['attack', 'attack_code']
        target_cols = 'attack_code'

    feature_cols = DF.drop(columns = col).columns

    if scaling:
      DF[feature_cols] =  scaleData(DF[feature_cols])

    return DF, feature_cols


def prepare_test_train_data(df_train, df_test, binary=True, category=True, scaling=True , target=True):
    """ Function to take prepare the input data for model prediction """
    df_train['train'] = 1
    df_test['train'] = 0
    df = pd.concat([df_train, df_test], ignore_index=True)
    df, feature_col = prepare_data(df, binary=binary, category=category, scaling=scaling , target=target)
    df_train = df[ df['train'] == 1]
    df_test = df[ df['train'] == 0]
    df.drop(['train'], axis=1, inplace=True)
    df_test.drop(['train'], axis=1, inplace=True)
    df_train.drop(['train'], axis=1, inplace=True)
    return df_train, df_test


def predict_data(model, X_test, Y_test):

    Y_predict       = model.predict(X_test)
    conf_mat        = confusion_matrix(Y_test, Y_predict)
    classfictin_rpt = classification_report(Y_test, Y_predict, output_dict=False)

    return Y_predict, conf_mat, classfictin_rpt


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

    target_cols = ['attack', 'attack_code']
    feature_cols = df.drop(columns = target_cols).columns

    df_train_scaled = scaleData( df[feature_cols])
    df_test_scaled  = scaleData( df_test[feature_cols])

    imp_cols = compute_important_cols(df_train, feature_cols, min_importance_value = 0.02, plot=False)

    # dump  test and train data into csv file
    df_train_scaled[feature_cols].to_csv('train_data.csv', header = True , index=False , columns=dump_df.columns )
    df_test_scaled[feature_cols].to_csv('test_data.csv', header = True , index=False , columns=dump_df.columns )
