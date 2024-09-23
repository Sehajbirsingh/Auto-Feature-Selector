import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectFromModel, chi2, SelectKBest
from sklearn.linear_model import LogisticRegression


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as ss
from collections import Counter
import math
from scipy import stats
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from lightgbm import LGBMClassifier

def cor_selector(X, y,num_feats):
    # Your code goes here (Multiple lines)
    cor_list = []
    feature_name = X.columns.tolist()
    for i in feature_name:
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append(cor)
    #print(np.argsort(np.abs(cor_list)))
    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()
    #print(cor_feature)
    cor_support = [True if i in cor_feature else False for i in feature_name]
    # Your code ends here
    return cor_support, cor_feature

def chi_squared_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    X_norm = MinMaxScaler().fit_transform(X)
    chi_selector = SelectKBest(chi2, k=num_feats)
    chi_selector.fit(X_norm, y)
    chi_support = chi_selector.get_support()
    #print(chi_support)
    chi_feature = X.loc[:,chi_support].columns.tolist()
    # Your code ends here
    return chi_support, chi_feature

def rfe_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    X_norm = MinMaxScaler().fit_transform(X)
    rfe_selector = RFE(estimator=LogisticRegression(random_state=42), n_features_to_select=num_feats, step=10, verbose=5)
    rfe_selector.fit(X_norm, y)
    rfe_support = rfe_selector.support_
    rfe_feature = X.loc[:,rfe_support].columns.tolist()
    # Your code ends here
    return rfe_support, rfe_feature

def embedded_log_reg_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    X_norm = MinMaxScaler().fit_transform(X)
    embedded_lr_selector = SelectFromModel(LogisticRegression(penalty="l2", random_state = 42), max_features=num_feats)
    embedded_lr_selector.fit(X_norm, y)
    embedded_lr_support = embedded_lr_selector.get_support()
    embedded_lr_feature = X.loc[:,embedded_lr_support].columns.tolist()
    # Your code ends here
    return embedded_lr_support, embedded_lr_feature

def embedded_rf_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    embeded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), max_features=num_feats)
    embeded_rf_selector.fit(X, y)
    embedded_rf_support = embeded_rf_selector.get_support()
    embedded_rf_feature = X.loc[:,embedded_rf_support].columns.tolist()
    # Your code ends here
    return embedded_rf_support, embedded_rf_feature

def embedded_lgbm_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    lgbc = LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=32, colsample_bytree=0.2, reg_alpha=3, reg_lambda=1, min_split_gain=0.01, min_child_weight=40,verbose=-1)
    embeded_lgbm_selector = SelectFromModel(lgbc, max_features=num_feats)
    embeded_lgbm_selector.fit(X, y)
    embedded_lgbm_support = embeded_lgbm_selector.get_support()
    embedded_lgbm_feature = X.loc[:,embedded_lgbm_support].columns.tolist()
    # Your code ends here
    return embedded_lgbm_support, embedded_lgbm_feature

def preprocess_dataset(dataset_path):
    # Load the dataset
    player_df = pd.read_csv(dataset_path)
    
    # Define numeric and categorical columns
    numcols = [
        'Overall', 'Crossing', 'Finishing', 'ShortPassing', 'Dribbling', 
        'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed', 
        'Agility', 'Stamina', 'Volleys', 'FKAccuracy', 'Reactions', 
        'Balance', 'ShotPower', 'Strength', 'LongShots', 'Aggression', 
        'Interceptions'
    ]
    catcols = ['Preferred Foot', 'Position', 'Body Type', 'Nationality', 'Weak Foot']
    
    # Filter the DataFrame to include only the specified columns
    player_df = player_df[numcols + catcols]

    # Create the training DataFrame with dummy variables for categorical features
    traindf = pd.concat([player_df[numcols], pd.get_dummies(player_df[catcols])], axis=1)
    
    # Drop rows with missing values
    traindf = traindf.dropna()

    
    # Define target variable and features
    y = traindf['Overall'] >= 87
    X = traindf.drop(columns=['Overall'])
    
    # Set the number of features to select
    num_feats = 30

    return X, y, num_feats

def autoFeatureSelector(dataset_path, methods=[]):

    X, y, num_feats = preprocess_dataset(dataset_path)
    
    # Initialize lists to store results
    feature_name = X.columns.tolist()
    
    # Initialize support lists for each method
    cor_support = [False] * len(feature_name)
    chi_support = [False] * len(feature_name)
    rfe_support = [False] * len(feature_name)
    embedded_lr_support = [False] * len(feature_name)
    embedded_rf_support = [False] * len(feature_name)
    embedded_lgbm_support = [False] * len(feature_name)
    
    # Run every method and collect selected features
    if 'pearson' in methods:
        cor_support, cor_feature = cor_selector(X, y, num_feats)
    if 'chi-square' in methods:
        chi_support, chi_feature = chi_squared_selector(X, y, num_feats)
    if 'rfe' in methods:
        rfe_support, rfe_feature = rfe_selector(X, y, num_feats)
    if 'log-reg' in methods:
        embedded_lr_support, embedded_lr_feature = embedded_log_reg_selector(X, y, num_feats)
    if 'rf' in methods:
        embedded_rf_support, embedded_rf_feature = embedded_rf_selector(X, y, num_feats)
    if 'lgbm' in methods:
        embedded_lgbm_support, embedded_lgbm_feature = embedded_lgbm_selector(X, y, num_feats)
    
    # Combine all the above feature lists and count how many times each feature got selected
    feature_selection_df = pd.DataFrame({
        'Feature': feature_name,
        'Pearson': np.array(cor_support).astype(bool),  # Convert to boolean
        'Chi-2': chi_support.astype(bool),
        'RFE': rfe_support.astype(bool),
        'Logistics': embedded_lr_support.astype(bool),
        'Random Forest': embedded_rf_support.astype(bool),
        'LightGBM': embedded_lgbm_support.astype(bool)
    })
    
    # Count how many times each feature was selected by different methods
    feature_selection_df['Total'] = feature_selection_df.iloc[:, 1:].sum(axis=1)
    
    # Sort the features by 'Total' and 'Feature' columns
    feature_selection_df = feature_selection_df.sort_values(['Total', 'Feature'], ascending=False)
    
    # Re-index the DataFrame
    feature_selection_df.index = range(1, len(feature_selection_df) + 1)
    
    # Get the top features
    best_features = feature_selection_df.head()['Feature'].tolist()
    
    return best_features

best_features = autoFeatureSelector(dataset_path="fifa19.csv", methods=['pearson', 'chi-square', 'rfe', 'log-reg', 'rf', 'lgbm'])
print(best_features)