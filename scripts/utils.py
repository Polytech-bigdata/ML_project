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
    col_num = [X.columns.get_loc(col) for col in X_num.columns]
    col_cat = [X.columns.get_loc(col) for col in X_cat.columns]
    
    #analyse data properties
    if debugging:
        print(f"Number of rows and columns with numerical values : {X_num.shape}")
        print(f"Number of rows and columns with categorical values : {X_cat.shape}")
        print(f"y.shape = {y.shape}")
        print(f"X_num = {X_num}")
        print(f"X_cat = {X_cat}")
        print(f"Number of positive numerical values: {len(y[y==1]/y.shape[0])*100}")
        print(f"Number of negative numerical values: {len(y[y==0]/y.shape[0])*100}")
   
    return y, X_num, X_cat, labels

def imputer_variables(X_num, X_cat, debugging=False):

    for col_id in range(X_cat.shape[1]): # shape 1 = number of columns    
        unique_val, val_idx = np.unique(X_cat[:, col_id], return_inverse=True) 
        X_cat[:, col_id] = val_idx 
    imp_cat = SimpleImputer(missing_values=0, strategy='most_frequent')
    X_cat[:, :] = imp_cat.fit_transform(X_cat[:, :]) 

    #for numerical variables, we will replace missing values by the mean of the column
    X_num = X_num.astype(float) 
    imp_num = SimpleImputer(missing_values=np.nan, strategy='mean') 
    X_num_imput = imp_num.fit_transform(X_num)

    #encode the categorical variables with one hot encoding
    X_cat_filled = LabelEncoder().fit_transform(X_cat).toarray()

    print(f"X_cat_filled: {X_cat_filled}")
    print(f"X_num_imput: {X_num_imput}")

    #concatenate the categorical and numerical variables normalized
    newDataset = np.hstack((X_cat_filled, X_num_imput))
    print(f"newDataset: {newDataset}")

    return newDataset