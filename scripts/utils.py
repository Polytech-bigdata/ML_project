#import libraries
import pickle
from copy import deepcopy
import numpy as np
from time import time
from sklearn.feature_selection import SelectFromModel, SelectKBest
from sklearn.pipeline import Pipeline 
np.set_printoptions(threshold=10000,suppress=True) 
import pandas as pd 
import warnings 
import matplotlib.pyplot as plt 
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, make_scorer, precision_score, recall_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.impute import SimpleImputer
#from xgboost import XGBClassifier

#global variables
clfs = None


def load_heterogeneous_dataset(dataset_filepath, header=0, delimiter=',', predict_var=True, debugging=False):
    """
    Charge un dataset hétérogène à partir d'un fichier CSV.

    Args:
        dataset_filepath (str): Chemin vers le fichier CSV contenant le dataset.
        header (int, optional): Ligne à utiliser comme en-tête. Par défaut 0.
        delimiter (str, optional): Délimiteur utilisé dans le fichier CSV. Par défaut ','.
        predict_var (bool, optional): Indique si la dernière colonne est la variable à prédire. Par défaut True. Pour savoir si on est en apprentissage supervisé ou non.
        debugging (bool, optional): Indique si des informations de débogage doivent être affichées. Par défaut False.

    Returns:
        tuple: Contient les éléments suivants:
            - X (DataFrame): Les caractéristiques du dataset.
            - y (Series ou None): La variable à prédire si predict_var est True, sinon None.
            - X_num (DataFrame): Les colonnes numériques de X.
            - X_cat (DataFrame): Les colonnes catégorielles de X.
            - labels (Index): Les étiquettes des colonnes de X.
    """
    dataset = pd.read_csv(dataset_filepath, header=header, sep=delimiter)
    if predict_var:
        X = dataset.iloc[:, 2:-2] #not include target column(last one)
        y = dataset.iloc[:,-1]
    else:
        X = dataset.iloc[:, 2:]
        y = None
    labels = dataset.columns[2:-2] # on fait -2 car la dernière colonne est la variable à prédire et la colonne avant est vide

    X_num = X.select_dtypes(include=[np.number]) # select numerical columns
    X_cat = X.select_dtypes(include=[object]) # select categorical columns
    #get the indexes of the columns
    col_num = [X_num.columns.get_loc(col) for col in X_num.columns]
    col_cat = [X_cat.columns.get_loc(col) for col in X_cat.columns]
    
    #analyse data properties
    if debugging:
        print(f"Number of rows and columns with numerical values : {X_num.shape}")
        print(f"Number of rows and columns with categorical values : {X_cat.shape}")
        print(f"y.shape = {y.shape}")
        print(f"X_num = {X_num}")
        print(f"X_cat = {X_cat}")
        print(f"Number of positive numerical values: {(len(y[y==1])/y.shape[0])*100}%")
        print(f"Number of negative numerical values: {(len(y[y==0])/y.shape[0])*100}%")
   
    return X_num, X_cat, y, labels


def imputer_variables(X_num, X_cat, debugging=False):
    # fill missing values in the categorical variables with most frequent value
    imp_cat = SimpleImputer(strategy='most_frequent')
    X_cat_filled = imp_cat.fit_transform(X_cat)
    X_cat_filled = pd.DataFrame(X_cat_filled, columns=X_cat.columns)
    
    #transform the dataset into a numpy array
    X_cat_filled = X_cat_filled.values
    X_cat_filled = X_cat_filled.astype(str)

    #transform the categorical variables into numerical variables
    enc = LabelEncoder()
    for i in range(X_cat_filled.shape[1]):
        X_cat_filled[:,i] = enc.fit_transform(X_cat_filled[:,i])
    X_cat_filled = X_cat_filled.astype(float)

    # fill missing values in the numerical variables with the mean of the column
    X_num = X_num.values
    X_num = X_num.astype(float) 
    imp_num = SimpleImputer(missing_values=np.nan, strategy='mean') 
    X_num_imput = imp_num.fit_transform(X_num)

    #concatenate the categorical and numerical variables to get the new dataset
    newDataset = np.hstack((X_cat_filled, X_num_imput))
    
    if debugging:
        print(f"X_cat_filled: {X_cat_filled}")
        print(f"X_num_imput: {X_num_imput}")
        print(f"newDataset: {newDataset}")

    return newDataset


def get_data_by_strategy(X, y, strategy: str = "natural", test_size=0.5, n_components=3, random_state=1):
    if strategy not in ["natural", "normalized", "pca"]:
        raise ValueError("Invalid strategy")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    if strategy != "natural":

        if strategy == "normalized" or strategy == "pca":
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        if strategy == "pca":
            pca = PCA(n_components=n_components)
            X_train_pca = pca.fit_transform(X_train)
            X_test_pca = pca.transform(X_test)
            X_train = np.hstack((X_train, X_train_pca))
            X_test = np.hstack((X_test, X_test_pca))

    return X_train, X_test, y_train, y_test


def scoring(y_test, y_pred):
    """
    Calculate the average recall score for both positive and negative classes.
    Parameters:
    y_test (array-like): True labels.
    y_pred (array-like): Predicted labels.
    Returns:
    float: The average recall score for both classes.
    """

    return (recall_score(y_test, y_pred, pos_label=0) + recall_score(y_test, y_pred, pos_label=1)) / 2


def run_classifiers(X, y, clfs, score_methode, n_splits=10, debugging=False):
    mean_scores = {}
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0) 
    for i in clfs:     
        clf = clfs[i]

        if debugging:
            print(f"Running {i}...")

        start = time()
 
        # Cross-validation for the criterion choosen: the call to the function scoring which
        # calculate the mean of the accuracy and the precision or the recall or the string name of the criterion available in the cross_val_score function (such as 'roc_auc')
        cv_crit = cross_val_score(clf, X, y, cv=kf, scoring=score_methode, n_jobs=-1) # cette méthode permet de faire de la cross validation

        mean_scores[i] = np.mean(cv_crit) # 
        
        execution_time = time() - start

        if debugging:
            print(f"Results for {i} :")
            print(f"Execution time: {execution_time:.3f} s")
            print(f"Mean of criterion choose for {i} is: {np.mean(cv_crit):.3f} +/- {np.std(cv_crit):.3f}")
            print("\n")

    nameBestModel = max(mean_scores, key=mean_scores.get)
    bestModel = clfs[ nameBestModel ]
    bestModelScore = mean_scores[ nameBestModel ]  
    if debugging:
        print(f"Best model: {bestModel} with score: {bestModelScore}")

    return bestModelScore, bestModel


def apply_ACP(X_scaled, n_components=3):
    pca = PCA(n_components=n_components,random_state=1)
    X_pca = pca.fit_transform(X_scaled)
    X_scaled = np.hstack((X_scaled, X_pca))
    return X_pca


def normalize_data(X):
    """
    Normalize the feature data using StandardScaler.
    Parameters:
    X_train (array-like): Training feature data.
    X_test (array-like): Test feature data.
    Returns:
    tuple: A tuple containing the normalized training and test feature data.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled
            

def comparison_cross_validation(X, y, clfs, scoring=scoring, n_splits=10, debugging=False):
    #case 1: without PCA
    #subcase 1: without normalization
    mean = make_scorer(scoring, greater_is_better=True)

    score, clf = run_classifiers(X, y, clfs, score_methode=mean, n_splits=n_splits, debugging=debugging)
    score_without_normalization = score
    clf_without_normalization = clf
    strategy = "natural"
    
    #subcase 2: with normalization
    X_scaled = normalize_data(X)
    score_normalized, clf_normalized = run_classifiers(X_scaled, y, clfs, score_methode=mean, n_splits=n_splits, debugging=debugging)
    
    # Get best model between model with normalization and model without normalization
    if score < score_normalized :
        score = score_normalized
        clf = clf_normalized
        strategy = "normalized" 
    
    #case 2: with PCA
    X_pca = apply_ACP(X_scaled)
    score_pca, clf_pca = run_classifiers(X_pca, y, clfs, score_methode=mean, n_splits=n_splits, debugging=debugging)
    
    #Finally, select the best model between the best model without PCA and the best model with PCA
    if score < score_pca :
        score = score_pca
        clf = clf_pca
        strategy = "pca"

    X_train_final, X_test_final, y_train, y_test = get_data_by_strategy(X, y, strategy) 

    if debugging:
        print(f"Finally with {n_splits} splits during cross-validation:")
        print(f"Best model without PCA and without normalization: {clf_without_normalization} with score: {score_without_normalization}")
        print(f"Best model without PCA and with normalization: {clf_normalized} with score: {score_normalized}")
        print(f"Best model with PCA and with normalization: {clf_pca} with score: {score_pca}")
        print(f"Finally the best model is: {clf} with score: {score} and strategy: {strategy}")

    return clf, X_train_final, X_test_final, y_train, y_test, strategy


def init_clfs(N_ESTIMATORS = 200, RANDOM_STATE = 1, N_NEIGHBORS = 5, N_COMPONENTS = 3):
    #global clfs
    clfs = {
        'RF': RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE),  # Random Forest
        'KNN': KNeighborsClassifier(n_neighbors=N_NEIGHBORS),  # K-Nearest Neighbors
        'MLP': MLPClassifier(hidden_layer_sizes=[20,10], random_state=RANDOM_STATE),  # Multi-Layer Perceptron
        'Naive Bayes': GaussianNB(),  # Naive Bayes
        'CART': DecisionTreeClassifier(random_state=RANDOM_STATE),  # Arbre CART
        'ID3': DecisionTreeClassifier(criterion='entropy', random_state=RANDOM_STATE),  # Arbre ID3
        'DS': DecisionTreeClassifier(max_depth=1, random_state=RANDOM_STATE),  # Decision Stump
        #'Bagging': BaggingClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE),  # Bagging
        'AdaBoost': AdaBoostClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE),  # AdaBoost
        #'SVM': SVC(random_state=RANDOM_STATE),  # Support Vector Machine
        #'XGBoost': XGBClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)  # XGBoost
    }
    return clfs

def init_clfs_parameters():
    clfs_parameters = {
        RandomForestClassifier: {
            'n_estimators': [200, 500, 1000],
            'max_features': ['auto', 'sqrt', 'log2']
        },
        KNeighborsClassifier: {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance']
        },
        MLPClassifier: {
            'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
            'activation': ['tanh', 'relu'],
            'solver': ['sgd', 'adam'],
            'alpha': [0.0001, 0.05],
            'learning_rate': ['constant','adaptive']
        },
        GaussianNB: {},
        DecisionTreeClassifier: {
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random']
        },
        # BaggingClassifier: {
        #     'n_estimators': [10, 100, 1000],
        #     'max_samples': [0.5, 1.0],
        #     'max_features': [0.5, 1.0]
        # },
        AdaBoostClassifier: {
            'n_estimators': [50, 100, 500],
            'learning_rate': [0.01, 0.1, 1]
        },
        # SVC: {
        #     'C': [1, 10, 100],
        #     'gamma': [0.1, 0.01, 0.001],
        #     'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
        # },
        # XGBClassifier: {
        #     'n_estimators': [100, 500, 1000],
        #     'max_depth': [3, 5, 7],
        #     'learning_rate': [0.01, 0.1, 1]
        # }
    }
    return clfs_parameters


def feature_importance(X_train, y_train, labels, debugging=False):
    clf = RandomForestClassifier(n_estimators=1000, random_state=1) 
    clf.fit(X_train, y_train) 
    importances=clf.feature_importances_ 

    std = np.std([tree.feature_importances_ for tree in clf.estimators_],axis=0) 
    
    sorted_idxs = np.argsort(importances)[::-1] 
    sorted_idx = [i for i in sorted_idxs if i < len(labels)]
    features = labels

    padding = np.arange(X_train.size/len(X_train)) + 0.5  
    padding = padding[:len(labels)]#take only number of features in padding present in labels (not those added by PCA)
    
    if debugging:
        print(f"Importances values of variables: \n {importances}")
        print(f"Labels sorted according to their importances : \n {features[sorted_idx].values}")
        plt.barh(padding, importances[sorted_idx], xerr=std[sorted_idx], align='center')  
        plt.yticks(padding, features[sorted_idx])  
        plt.xlabel("Relative Importance") 
        plt.title("Variable Importance")  
        plt.show() 

    return sorted_idx
    

def feature_selection(X_train, X_test, y_train, y_test, clf, sorted_idx, scoring=scoring, score_type='Rec+_Rec-', debugging=False):
    nb_total_features = X_train.shape[1]+1
    scores = np.zeros(nb_total_features)
    nb_selected_features = 0
    max_score = 0
    for f in np.arange(0, nb_total_features):  
        X1_f = X_train[:,sorted_idx[:f+1]] 
        X2_f = X_test[:,sorted_idx[:f+1]] 
        clf.fit(X1_f,y_train) 
        output = clf.predict(X2_f) 

        mean = scoring(y_test,output) 
        scores[f] = np.round(mean,3) 
        
        if max_score < scores[f]:
            nb_selected_features = f
            max_score = scores[f]

    if debugging:
        print(f"Number of features selected : {nb_selected_features}")
        plt.plot(scores) 
        plt.xlabel("Nombre de Variables") 
        plt.ylabel("({} / 2)".format(score_type))
        plt.title("Evolution en fonction des variables") 
        plt.show()

    return nb_selected_features


#function using the GridSearchCV function in the bestModel found previously
def fine_tune_model(X_train, y_train, bestModel, param_grid, scoring=scoring, debugging=False):
    #makeScorer use for gridSearchCV
    scoring = make_scorer(scoring, greater_is_better=True)

    grid_search = GridSearchCV(bestModel, param_grid, n_jobs=-1, cv=5, scoring=scoring)
    
    grid_search.fit(X_train, y_train) 

    best_score = grid_search.best_score_
    best_model = grid_search.best_estimator_
    
    if debugging:
        print(f"Best score after parameters: {best_score}")
        print(f"The best fine-tuned model is : \n {best_model}")
        
    return best_model

def create_pickle_file(steps, pipeline_filepath):
    pipeline = Pipeline(steps)
    with open(pipeline_filepath, "wb") as file:
        pickle.dump(pipeline, file)
        print(f"Pipeline saved as {pipeline_filepath}")
    return pipeline


def creation_pipelines(X, y, model, strategy, nb_features, artifacts_path= '../artifacts/', debugging=False):

    # Create the pipelines for Scaler, PCA, Feature selection and Classifier

    # Create the pipeline for Scaler
    steps = []
    if strategy == "normalized" or strategy == "PCA":
        steps.append(("scaler", StandardScaler()))
        scaler_pipeline = create_pickle_file(steps, artifacts_path + "scaler.pkl")
        X, y = scaler_pipeline.fit_transform(X, y)
        if debugging:
            print(f"Pipeline created: {scaler_pipeline}")
        
    # Create the pipeline for PCA
    steps = []
    if strategy == "pca":
        steps.append(("pca", PCA(n_components=3)))
        pca_pipeline = create_pickle_file(steps, artifacts_path + "pca.pkl")
        X, y = pca_pipeline.fit_transform(X, y)
        if debugging:
            print(f"Pipeline created: {pca_pipeline}")

    # Create the pipeline for Feature selection and Classifier
    steps = []
    steps.append(("fs", SelectFromModel(RandomForestClassifier(n_estimators=1000, random_state=1), max_features=nb_features)))
    steps.append(("classifier", model))

    pipeline = create_pickle_file(steps, artifacts_path + "model.pkl")

    pipeline.fit(X, y)

    if debugging:
        print(f"Pipeline created: {pipeline}")


def load_pipeline(pipeline_filepath):
    with open(pipeline_filepath, "rb") as file:
        pipeline = pickle.load(file)
    return pipeline


def learning(dataset_filepath, clfs, clfs_parameters, comparison_func=comparison_cross_validation ,criterion=scoring, debugging=False):
    # load the dataset
    X_num, X_cat, y, labels = load_heterogeneous_dataset(dataset_filepath)

    # impute the missing values
    X = imputer_variables(X_num, X_cat, debugging=debugging)

    # get the best model, new X_train and X_test (normalized or not, columns added by PCA) and strategy
    model, X_train, X_test, y_train, y_test, strategy = comparison_func(X, y, clfs, scoring=criterion, debugging=debugging)

    # get the most important features
    sorted_idx = feature_importance(X_train, y_train, labels, debugging=debugging)

    # select the most important features
    nb_selected_features = feature_selection(X_train, X_test, y_train, y_test, model, sorted_idx, scoring=criterion, debugging=debugging)

    #select for this model the corresponding parameters grid
    param_grid = clfs_parameters[type(model)] 

    #update X_train and X_test with the selected features
    X_train = X_train[:,sorted_idx[:nb_selected_features]]

    # fine-tune the model
    best_model = fine_tune_model(X_train, y_train, model, param_grid, scoring=criterion, debugging=debugging)

    # create the pipeline
    creation_pipelines(X, y, best_model, strategy, nb_selected_features, debugging=debugging)

    #load the pipeline
    pipeline = load_pipeline("pipeline.pkl")

    print(f"End of process")
    return pipeline


def update_labels_for_stragegy(labels, strategy, nb_components=3):
    if strategy == "normalized":
        labels =[label +"_normalized" for label in labels]
    if strategy == "pca":
        labels = np.hstack((labels, [f"PCA_{i}" for i in range(nb_components)]))
    # Add the target column to the labels
    labels = np.hstack((labels, ["target"]))
    print(f"Updated labels : {labels}")
    return labels


def create_data_csv(X_list, y_list, labels, csv_filename, csv_filepath="../data/"):
    if len(X_list) != len(y_list):
        raise ValueError("X_list and y_list must have the same length")
    X=[]
    y=[]
    if len(X_list) == 2:
        # Concatenate the two datasets and targets
        for i in range(len(X_list)):
            for j in range(len(X_list[i])):
                X.append(X_list[i][j])
                y.append(y_list[i][j])
    else:
        X = X_list[0]
        y = y_list[0]
    #transform into True and False the target column
    y = np.array(y)
    y = y.astype(bool)
    print(f"y as booleans={y}")
    # Add the target column to the dataframe
    data = np.hstack((X, y.reshape(-1, 1)))
    df = pd.DataFrame(data,columns=labels)
    df.to_csv(csv_filepath + csv_filename,sep= ";", index=False)
    print(f"Data saved in {csv_filepath + csv_filename}")