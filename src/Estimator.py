# Helper Class for Initilizing GridSearch
from sklearn.model_selection import GridSearchCV


# name of the columns of the data frame which captures all the data of all algorithms
MLA_Name = 'MLA Name'
MLA_Param = 'MLA Parameters'
MLA_Train_Accuracy = 'MLA Train Accuracy'
MLA_Validation_Accuracy = 'MLA Validation Accuracy'
MLA_Validation_STD = 'MLA Validation Accuracy 3*STD'
MLA_Test_Accuracy = 'MLA Test Accuracy'
MLA_Time = 'MLA Time'

class EstimatorSelectionHelper:
    """Class to train and evaluate and score differnt ML models.
    It also retuns important features , best parameters after ML model evaluation
    It also dumps to model to local file system so that it can be used for later use
    """

    def __init__(self):

        self.models = {}
        self.params = {}
        self.grid_searches = {}
        self.best_params = {}
        self.feature_importance = {}
        self.FeatureImportanceAlgo = ['DecisionTreeClassifier','RandomForestClassifier','ExtraTreesClassifier','GradientBoostingClassifier']
        self.MLA = pd.DataFrame(columns = [MLA_Name, MLA_Param, MLA_Time, MLA_Train_Accuracy, MLA_Validation_Accuracy, MLA_Test_Accuracy])


    def score(self, X_test, Y_test):
        """function scores all added ML models """
        df = self.MLA
        for k in self.grid_searches:
            print(k)
            algo = self.grid_searches[k]
            df.loc[ df[MLA_Name]== k , MLA_Test_Accuracy] = algo.score(X_test, Y_test)
        return self.MLA

    def fit(self, X, y, cv=3, n_jobs=3, verbose=1, scoring=None, refit=True):
        """function fits all added ML models """
        if not set(self.models.keys()).issubset(set(self.params.keys())):
            missing_params = list(set(self.models.keys()) - set(self.params.keys()))
            raise ValueError("Some estimators are missing parameters: %s" % missing_params)

        for key in self.models.keys():
            print("Running GridSearchCV for %s." % key)
            model = self.models[key]
            params = self.params[key]
            gs = GridSearchCV(model, params, cv=cv, n_jobs=n_jobs,
                              verbose=verbose, scoring=scoring, refit=refit,
                              return_train_score=True)
            gs.fit(X,y)
            self.grid_searches[key] = gs
            self.best_params[key]  = str(gs.best_params_)
            if key in self.FeatureImportanceAlgo:
                self.feature_importance[key]= gs.best_estimator_ .feature_importances_


            # print (gs.best_params_.feature_importances_ )
            # try:
            #   print(gs.best_params_.feature_importances_ )
            #   self.feature_importance[key]= gs.best_params_.feature_importances_
            # except AttributeError:
            #   pass

    def imp_features(self):
        """function returns the important feature evaluated after different ML evaluation"""

        d = self.feature_importance
        impDF = pd.DataFrame([d.keys(), d.values()])
        return impDF

    def returnBestParamDF(self):
        """function returns the best paramemtes evaluated after different ML evaluation"""
        d = self.best_params
        BestParamDF = pd.DataFrame.from_dict([d.keys(), d.values()]).T
        return BestParamDF

    def add_model_and_params(self, name, model, hyperparam):
        """function adds ML model and its Parameter to the class for evaluation """
        self.models[name] =  model
        self.params[name] = hyperparam

    def fit_summary(self):
        """function accumulates the scores of each ML model into a member Data Frame"""
        arr = []
        for k in self.grid_searches:
            dict= {}
            # print(k)
            algo = self.grid_searches[k]
            dict[MLA_Name] = k
            dict[MLA_Param] = str(algo.best_params_)
            dict[MLA_Time] = np.nanmean( algo.cv_results_['mean_fit_time'])
            dict[MLA_Train_Accuracy] = np.nanmean(algo.cv_results_['mean_train_score'])
            dict[MLA_Validation_Accuracy] = np.nanmean(algo.cv_results_['mean_test_score'], )
            dict[MLA_Test_Accuracy] = 0
            arr.append(dict)

        self.MLA = pd.DataFrame(arr)
        return self.MLA

    def save_model_to_file(self, modelname, filename='model.mdl'):
        """ function saves the input ML model is saved as input file """
        if modelname not in self.grid_searches.keys():
            print("Model doesn't not exist !!  No file is saved")
            return False
        with open(filename, "wb") as file:
            pickle.dump(self.grid_searches[modelname],file)
            print("Model {} saved into file {}".format(modelname, filename))
        return True
