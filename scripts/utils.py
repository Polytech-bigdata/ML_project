#import libraries
import pickle
from copy import deepcopy
import numpy as np
from time import time
from sklearn import set_config
from sklearn.compose import ColumnTransformer
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
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.impute import SimpleImputer
#from xgboost import XGBClassifier
import os

#global variables
clfs = None

def load_heterogeneous_dataset(dataset_filepath, header=0, delimiter=',', predict_var=True, debugging=False):
    """
    Load a heterogeneous dataset from a CSV file, separating numerical and categorical features.
    
    Parameters:
    dataset_filepath (str): The path to the CSV file containing the dataset.
    header (int or None): Row number to use as the column names. Default is 0.
    delimiter (str): The delimiter to use for separating values in the CSV file. Default is ','.
    predict_var (bool): Whether the dataset includes a target variable to predict. Default is True.
    debugging (bool): If True, prints debugging information about the dataset. Default is False.
    
    Returns:
    tuple: A tuple containing:
        - X (DataFrame): The feature matrix.
        - y (Series or None): The target variable if predict_var is True, otherwise None.
        - col_num (list): List of indices of numerical columns in X.
        - col_cat (list): List of indices of categorical columns in X.
        - labels (Index): Column labels for the features.
    """
    dataset = pd.read_csv(dataset_filepath, header=header, sep=delimiter)
    if predict_var:
        # select variables excluding the 2 first columns (encounter_id and patient_id) 
        # and the target column with the previous (because always empty in the original dataset)
        X = dataset.iloc[:, 2:-2]
        y = dataset.iloc[:,-1]
    else:
        X = dataset.iloc[:, 2:]
        y = None
    labels = dataset.columns[2:-2]
    
    # select numerical columns
    X_num = X.select_dtypes(include=[np.number]) 
    # select categorical columns
    X_cat = X.select_dtypes(include=[object]) 

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
        print(f"Number of positive numerical values: {(len(y[y==1])/y.shape[0])*100}%")
        print(f"Number of negative numerical values: {(len(y[y==0])/y.shape[0])*100}%")
   
    return  X, y, col_num, col_cat, labels


def imputer_variables(X, col_num, col_cat, debugging=False):
    """
    Impute missing values in numerical and categorical variables and transform categorical variables to numerical.
    
    Parameters:
    X (pd.DataFrame): The input dataframe containing both numerical and categorical variables.
    col_num (list): List of column indices for numerical variables.
    col_cat (list): List of column indices for categorical variables.
    debugging (bool): If True, prints intermediate steps for debugging purposes. Default is False.
    
    Returns:
    np.ndarray: A new dataset with imputed and transformed variables.
    """
    X_cat = X.values[:, col_cat]
    X_num = X.values[:, col_num]

    # fill missing values in the categorical variables with most frequent value
    imp_cat = SimpleImputer(strategy='most_frequent')
    X_cat_filled = imp_cat.fit_transform(X_cat)
    X_cat_filled = X_cat_filled.astype(str)

    #transform the categorical variables into numerical variables
    enc = LabelEncoder()
    for i in range(X_cat_filled.shape[1]):
        X_cat_filled[:,i] = enc.fit_transform(X_cat_filled[:,i])
    X_cat_filled = X_cat_filled.astype(float)

    # fill missing values in the numerical variables with the mean of the column
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
    """
    Splits the dataset into training and testing sets and applies the specified preprocessing strategy.
    
    Parameters:
    X (pd.DataFrame or np.ndarray): Features dataset.
    y (pd.Series or np.ndarray): Target labels.
    strategy (str, optional): Preprocessing strategy to apply. Options are "natural", "normalized", and "pca". Default is "natural".
    test_size (float, optional): Proportion of the dataset to include in the test split. Default is 0.5.
    n_components (int, optional): Number of principal components to keep if strategy is "pca". Default is 3.
    random_state (int, optional): Controls the shuffling applied to the data before applying the split. Default is 1.
    
    Returns:
    tuple: A tuple containing four elements:
        - X_train (np.ndarray): Training features.
        - X_test (np.ndarray): Testing features.
        - y_train (np.ndarray): Training labels.
        - y_test (np.ndarray): Testing labels.
    
    Raises:
    ValueError: If the provided strategy is not one of "natural", "normalized", or "pca".
    """
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
    
    #convert into numpy array y_train and y_test
    y_train = y_train.values
    y_test = y_test.values

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
    """
    Run multiple classifiers with cross-validation and return the best model and its score.
    
    Parameters:
    X : array-like or sparse matrix of shape (n_samples, n_features)
        The input data to fit.
    y : array-like of shape (n_samples,)
        The target variable to try to predict.
    clfs : dict
        A dictionary where keys are classifier names and values are classifier instances.
    score_methode : str
        The scoring method to use for cross-validation (e.g., 'accuracy', 'precision', 'recall', 'roc_auc').
    n_splits : int, optional (default=10)
        Number of folds for cross-validation.
    debugging : bool, optional (default=False)
        If True, print debugging information.
    
    Returns:
    bestModelScore : float
        The best score obtained from cross-validation.
    bestModel : estimator object
        The classifier instance that achieved the best score.
    """
    mean_scores = {}
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0) 
    for i in clfs:     
        clf = clfs[i]

        if debugging:
            print(f"Running {i}...")

        start = time()
 
        # Cross-validation for the criterion choosen: the call to the scoring function  which
        # calculate the mean of the accuracy and the precision or the recall or the string name 
        # of the criterion available in the cross_val_score function (such as 'roc_auc')
        cv_crit = cross_val_score(clf, X, y, cv=kf, scoring=score_methode, n_jobs=-1)

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
    """
    Apply Principal Component Analysis (PCA) to the scaled data.
    
    Parameters:
    X_scaled (numpy.ndarray): The scaled input data.
    n_components (int): The number of principal components to compute. Default is 3.
    
    Returns:
    numpy.ndarray: The transformed data with the principal components.
    """
    pca = PCA(n_components=n_components,random_state=1)
    X_pca = pca.fit_transform(X_scaled)
    X_scaled = np.hstack((X_scaled, X_pca))
    return X_pca


def normalize_data(X):
    """
    Normalize the input data using StandardScaler.
    
    Parameters:
    X (array-like): The input data to be normalized.
    
    Returns:
    array-like: The normalized data.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled
            

def comparison_cross_validation(X, y, clfs, scoring=scoring, n_splits=10, debugging=False):
    """
    Perform cross-validation to compare classifiers with different preprocessing strategies.
    This function evaluates classifiers using cross-validation with three different preprocessing strategies:
    1. Without PCA and without normalization.
    2. Without PCA and with normalization.
    3. With PCA and with normalization.
    The function selects the best model based on the scoring metric provided and returns the best classifier along with the training and testing datasets.
    
    Parameters:
    X (array-like): Feature matrix.
    y (array-like): Target vector.
    clfs (list): List of classifiers to evaluate.
    scoring (callable): Scoring function to evaluate the classifiers.
    n_splits (int, optional): Number of splits for cross-validation. Default is 10.
    debugging (bool, optional): If True, prints debugging information. Default is False.
    
    Returns:
    tuple: A tuple containing:
        - clf (estimator): The best classifier.
        - X_train_final (array-like): Training feature matrix based on the best strategy.
        - X_test_final (array-like): Testing feature matrix based on the best strategy.
        - y_train (array-like): Training target vector.
        - y_test (array-like): Testing target vector.
        - strategy (str): The best preprocessing strategy used ("natural", "normalized", or "pca").
    """
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
        #'RF': RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE),  # Random Forest
        #'KNN': KNeighborsClassifier(n_neighbors=N_NEIGHBORS),  # K-Nearest Neighbors
        #'MLP': MLPClassifier(hidden_layer_sizes=[20,10], random_state=RANDOM_STATE),  # Multi-Layer Perceptron
        'Naive Bayes': GaussianNB(),  # Naive Bayes
        #'CART': DecisionTreeClassifier(random_state=RANDOM_STATE),  # Arbre CART
        #'ID3': DecisionTreeClassifier(criterion='entropy', random_state=RANDOM_STATE),  # Arbre ID3
        #'DS': DecisionTreeClassifier(max_depth=1, random_state=RANDOM_STATE),  # Decision Stump
        #'Bagging': BaggingClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE),  # Bagging
        #'AdaBoost': AdaBoostClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE),  # AdaBoost
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
    """
    Compute and plot the feature importance using a RandomForestClassifier.
    
    Parameters:
    X_train (pd.DataFrame or np.ndarray): Training data features.
    y_train (pd.Series or np.ndarray): Training data labels.
    labels (pd.Index or list): List of feature names.
    debugging (bool): If True, prints debugging information and plots feature importances. Default is False.
    
    Returns:
    list: Indices of features sorted by importance in descending order.
    """
    clf = RandomForestClassifier(n_estimators=1000, random_state=1) 
    clf.fit(X_train, y_train) 
    importances=clf.feature_importances_ 

    std = np.std([tree.feature_importances_ for tree in clf.estimators_],axis=0) 
    
    sorted_idxs = np.argsort(importances)[::-1] 
    sorted_idx = [i for i in sorted_idxs if i < len(labels)]
    features = labels

    #for plotting the importances values add space between the bars
    padding = np.arange(X_train.size/len(X_train)) + 0.5  

    #take only number of features in padding present in labels (not those added by PCA)
    padding = padding[:len(labels)]
    
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
    """
    Perform feature selection by iteratively adding features and evaluating the model performance.
    
    Parameters:
    X_train (numpy.ndarray): Training data features.
    X_test (numpy.ndarray): Test data features.
    y_train (numpy.ndarray): Training data labels.
    y_test (numpy.ndarray): Test data labels.
    clf (object): Classifier object with fit and predict methods.
    sorted_idx (numpy.ndarray): Indices of features sorted by importance.
    scoring (callable): Scoring function to evaluate model performance.
    score_type (str, optional): Type of score to display in the plot title. Default is 'Rec+_Rec-'.
    debugging (bool, optional): If True, prints debugging information and plots the score evolution. Default is False.
    
    Returns:
    int: Number of selected features that resulted in the highest score.
    """
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


def fine_tune_model(X_train, y_train, bestModel, param_grid, scoring=scoring, debugging=False):
    """
    Fine-tunes a given model using GridSearchCV with the provided parameter grid and scoring function.
    
    Parameters:
    X_train (array-like or sparse matrix): The training input samples.
    y_train (array-like): The target values (class labels) as integers or strings.
    bestModel (estimator object): The base model to be fine-tuned.
    param_grid (dict or list of dictionaries): Dictionary with parameters names (str) as keys and lists of parameter settings to try as values, or a list of such dictionaries.
    scoring (callable): A scoring function to evaluate the predictions on the test set. It should be a callable that returns a single value.
    debugging (bool, optional): If True, prints the best score and the best model after fine-tuning. Default is False.
    
    Returns:
    estimator object: The best fine-tuned model found by GridSearchCV.
    """
    #transform the scoring function with makescorer in order to use it for GridSearchCV
    scoring = make_scorer(scoring, greater_is_better=True)

    grid_search = GridSearchCV(bestModel, param_grid, n_jobs=-1, cv=5, scoring=scoring)
    
    grid_search.fit(X_train, y_train) 

    best_score = grid_search.best_score_
    best_model = grid_search.best_estimator_
    
    if debugging:
        print(f"Best score after parameters: {best_score}")
        print(f"The best fine-tuned model is : \n {best_model}")
        
    return best_model


def create_pickle_file(pipeline, pipeline_filepath):
    """
    Serializes a machine learning pipeline object and saves it to a specified file path using pickle.
    Args:
        pipeline (object): The machine learning pipeline object to be serialized.
        pipeline_filepath (str): The file path where the serialized pipeline will be saved.
    Returns:
        None
    """
    with open(pipeline_filepath, "wb") as file:
        pickle.dump(pipeline, file)
        print(f"Pipeline saved as {pipeline_filepath}")


def creation_pipelines(X, y, num_col, cat_col, model, strategy, nb_features, artifacts_path= '../artifacts/', debugging=False):
    """
    Create the pipelines for Scaler, PCA, Feature selection and Classifier based on the provided strategy.
    
    Parameters:
    X (pd.DataFrame): The input features.
    y (pd.Series): The target variable.
    num_col (list): List of numerical columns.
    cat_col (list): List of categorical columns.
    model (sklearn.base.BaseEstimator): The machine learning model to be used.
    strategy (str): The strategy for preprocessing ('normalized', 'pca', etc.).
    nb_features (int): Number of features to select.
    artifacts_path (str, optional): Path to save the pipeline artifacts. Defaults to '../artifacts/'.
    debugging (bool, optional): If True, prints debugging information. Defaults to False.
    
    Returns:
    None
    """
    #for all pipelines display the diagram of the pipeline
    set_config(display="diagram")

    # Create the pipeline for Imputer
    numerical_transformer = SimpleImputer(missing_values=np.nan, strategy='mean')
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder())
    ])
    
    preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, num_col),
        ('cat', categorical_transformer, cat_col)
    ])
    
    imputer_pipeline = Pipeline(steps=[("preprocessor", preprocessor)])
    imputer_pipeline.fit(X)
    X = imputer_pipeline.transform(X)
    create_pickle_file(imputer_pipeline, artifacts_path + "imputer.pkl")
    if debugging:
        print(f"Pipeline created: {imputer_pipeline}")
                    
    # Create the pipeline for Scaler
    if strategy == "normalized" or strategy == "pca":
        scaler_pipeline = Pipeline(steps=[("scaler", StandardScaler())])
        X, y = scaler_pipeline.fit_transform(X, y)
        create_pickle_file(scaler_pipeline, artifacts_path + "scaler.pkl")
        if debugging:
            print(f"Pipeline created: {scaler_pipeline}")
        
    # Create the pipeline for PCA
    if strategy == "pca":
        pca_pipeline = Pipeline(steps=[("pca", PCA(n_components=3))])
        X, y = pca_pipeline.fit_transform(X, y)
        create_pickle_file(pca_pipeline, artifacts_path + "pca.pkl")
        if debugging:
            print(f"Pipeline created: {pca_pipeline}")

    # Create the pipeline for Feature selection and Classifier
    steps = []
    steps.append(("fs", SelectFromModel(RandomForestClassifier(n_estimators=1000, random_state=1, coef_= model.feature_importances_), max_features=nb_features)))
    steps.append(("classifier", model))

    model_pipeline = Pipeline(steps)
    model_pipeline.fit(X, y)
    create_pickle_file(model_pipeline, artifacts_path + "model.pkl")
    
    if debugging:
        print(f"Pipeline created: {model_pipeline}")


def load_pipeline(pipeline_filepath) -> Pipeline :
    """
    Load a machine learning pipeline from a file.
    Args:
        pipeline_filepath (str): The file path to the pipeline file.
    Returns:
        Pipeline: The loaded machine learning pipeline object if successful, 
                    otherwise None if an error occurs during loading.
    """
    try:
        with open(pipeline_filepath, "rb") as file:
            pipeline = pickle.load(file)
        return pipeline
    except:
        return None


def learning(dataset_filepath, clfs, clfs_parameters, comparison_func=comparison_cross_validation ,criterion=scoring, debugging=False):
    # load the dataset
    X, y, col_num, col_cat, labels = load_heterogeneous_dataset(dataset_filepath)
 
    # impute the missing values
    X_concat = imputer_variables(X, col_num, col_cat, debugging=debugging)

    # get the best model, new X_train and X_test (normalized or not, columns added by PCA) and strategy
    model, X_train, X_test, y_train, y_test, strategy = comparison_func(X_concat, y, clfs, scoring=criterion, debugging=debugging)

    # get the most important features
    sorted_idx = feature_importance(X_train, y_train, labels, debugging=debugging)

    # select the most important features
    nb_selected_features = feature_selection(X_train, X_test, y_train, y_test, model, sorted_idx, scoring=criterion, debugging=debugging)

    #select for this model the corresponding parameters grid
    param_grid = clfs_parameters[type(model)] 

    #update X_train and X_test with the selected features
    X_train_selected = X_train[:,sorted_idx[:nb_selected_features]]

    # fine-tune the model
    best_model = fine_tune_model(X_train_selected, y_train, model, param_grid, scoring=criterion, debugging=debugging)

    # create the pipeline
    creation_pipelines(X_concat, y, col_num, col_cat, best_model, strategy, nb_selected_features, debugging=debugging)

    #update labels for the strategy
    labels = update_labels_for_stragegy(labels, strategy)

    # save the data imputed and scaled/with PCA in a csv file
    create_data_csv([X_train, X_test], [y_train, y_test], labels, 'ref_data.csv')

    print(f"End of process")


def update_labels_for_stragegy(labels, strategy, nb_components=3):
    """
    Update the list of labels based on the given strategy.
    Parameters:
    labels (list): List of initial labels.
    strategy (str): Strategy to update the labels. Can be "normalized" or "pca".
    nb_components (int, optional): Number of PCA components to add if strategy is "pca". Default is 3.
    Returns:
    list: Updated list of labels with the applied strategy and the target column added.
    """
    if strategy == "normalized":
        labels =[label +"_normalized" for label in labels]
    if strategy == "pca":
        labels = np.hstack((labels, [f"PCA_{i}" for i in range(nb_components)]))
    # Add the target column to the labels
    labels = np.hstack((labels, ["target"]))
    print(f"Updated labels : {labels}")
    return labels


def create_data_csv(X_list, y_list, labels, csv_filename, csv_filepath="../data/"):
    """
    Creates a CSV file from the provided datasets and labels.
    Parameters:
    X_list (list of arrays): List containing feature datasets.
    y_list (list of arrays): List containing target datasets.
    labels (list of str): List of column names for the CSV file.
    csv_filename (str): Name of the CSV file to be created.
    csv_filepath (str, optional): Path where the CSV file will be saved. Defaults to "../data/".
    Raises:
    ValueError: If the lengths of X_list and y_list do not match.
    Returns:
    None
    """
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
    # Add the target column to the dataframe
    data = np.hstack((X, y.reshape(-1, 1)))
    df = pd.DataFrame(data,columns=labels)
    df.to_csv(csv_filepath + csv_filename,sep= ";", index=False)
    print(f"Data saved in {csv_filepath + csv_filename}")


def pickle_file_exists():
    #depending if the scaler or pca pickle is detected in the artifacts folder, apply the transformation
    pipelines = []
    if "scaler.pkl" in os.listdir("artifacts"):
        scaler = load_pipeline("artifacts/scaler.pkl")
        pipelines.append(scaler)
    elif "pca.pkl" in os.listdir("artifacts"):
        pca = load_pipeline("artifacts/pca.pkl")
        pipelines.append(pca)
    return pipelines    