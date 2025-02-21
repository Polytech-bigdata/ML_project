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
from xgboost import XGBClassifier

clfs = None

def init_clfs(N_ESTIMATORS = 200, RANDOM_STATE = 1, N_NEIGHBORS = 5, N_COMPONENTS = 3):
    global clfs
    clfs = {
        'RF': RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE),  # Random Forest
        'KNN': KNeighborsClassifier(n_neighbors=N_NEIGHBORS),  # K-Nearest Neighbors
        'MLP': MLPClassifier(random_state=RANDOM_STATE),  # Multi-Layer Perceptron
        'NB': GaussianNB(),  # Naive Bayes
        'CART': DecisionTreeClassifier(random_state=RANDOM_STATE),  # Arbre CART
        'ID3': DecisionTreeClassifier(criterion='entropy', random_state=RANDOM_STATE),  # Arbre ID3
        'DS': DecisionTreeClassifier(max_depth=1, random_state=RANDOM_STATE),  # Decision Stump
        'Bagging': BaggingClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE),  # Bagging
        'AdaBoost': AdaBoostClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE),  # AdaBoost
        'XGBoost': XGBClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)  # XGBoost
    }

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
        print(f"Number of positive numerical values: {len(y[y==1]/y.shape[0])*100}")
        print(f"Number of negative numerical values: {len(y[y==0]/y.shape[0])*100}")
   
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size, random_state)
    if strategy != "natural":
        if strategy == "normalized" or strategy == "pca":
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        if strategy == "pca":
            pca = PCA(n_components=n_components)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)
    return X_train, X_test, y_train, y_test


def scoring(y_test, y_pred, score_type='Acc_Prec'):
    specific_score = precision_score
    if score_type == 'Acc_Rec':
        specific_score = recall_score
    elif score_type != 'Acc_Prec':
        raise ValueError(f"Unknown score type: {score_type}")
    return (accuracy_score(y_test, y_pred) + specific_score(y_test, y_pred)) / 2


def run_classifiers(X, y, clfs, score_methode, debugging=False):
    mean_scores = {}
    kf = KFold(n_splits=10, shuffle=True, random_state=0) 
    for i in clfs:     
        clf = clfs[i]

        start = time()
 
        # Cross-validation for the criterion choosen: the call to the function scoring which
        # calculate the mean of the accuracy and the precision or the recall or the string name of the criterion available in the cross_val_score function (such as 'roc_auc')
        cv_crit = cross_val_score(clf, X, y, cv=kf, scoring=score_methode, n_jobs=-1)

        mean_scores[i] = np.mean(cv_crit)
        
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
    
    X_train = np.hstack((X_train, X_pca))
    #X_test = np.hstack((X_test, X_test_pca))
    
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
            

def comparison_cross_validation(X, y, clfs, scoring=scoring, debugging=False):
    #case 1: without PCA
    #subcase 1: without normalization
    mean = make_scorer(scoring, greater_is_better=True)
    score, clf = run_classifiers(X, y, clfs, score_methode=mean, debugging=debugging)
    score_without_normalization = score
    clf_without_normalization = clf
    strategy = "natural"
    
    #subcase 2: with normalization
    X_scaled = normalize_data(X)
    score_normalized, clf_normalized = run_classifiers(X_scaled, y, clfs, score_methode=mean, debugging=debugging)
    
    # Get best model between model with normalization and model without normalization
    if score < score_normalized :
        score = score_normalized
        clf = clf_normalized
        strategy = "normalized" 
    
    #case 2: with PCA
    X_pca = apply_ACP(X_scaled, n_components=3)
    score_pca, clf_pca = run_classifiers(X_pca, y, clfs, score_methode=mean, debugging=debugging)
    
    #Finally, select the best model between the best model without PCA and the best model with PCA
    if score < score_pca :
        score = score_pca
        clf = clf_pca
        strategy = "pca"

    X_train_final, X_test_final, y_train, y_test =  get_data_by_strategy(X, y, strategy) 

    if debugging:
        print(f"Best model without PCA and without normalization: {clf_without_normalization} with score: {score_without_normalization}")
        print(f"Best model without PCA and with normalization: {clf_normalized} with score: {score_normalized}")
        print(f"Finally the best model is: {clf} with score: {score} and strategy: {strategy}")

    return clf, X_train_final, X_test_final, y_train, strategy


