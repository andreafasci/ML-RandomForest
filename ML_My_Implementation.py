
# coding: utf-8

# In[ ]:


# Import useful stuff from libraries etc
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import random

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator

np.random.seed(0)


# In[ ]:


# Define class MyRandomForest, inheriting BaseEstimator class (BaseEstimator is used only for CrossValidation purposes)
class MyRandomForest(BaseEstimator):
    
    # Class constructor, default values are here selected, but other values can be specified when creating the RF)
    def __init__(self, n_estimators=100, random_state=None, max_depth=None,
                 max_features=None, min_samples_leaf=1, min_samples_split=2,
                class_weight=None, criterion='gini'):
        # Create an array of Decision Trees
        self.__listOfDecisionTrees = []
        # For each Decision Tree, save which are the features used by this particular Tree
        self.__listOfFeaturesPerTree = []
        # Save the number of estimators
        self.__n_estimators = n_estimators
        
        # Save other parameters for trees
        self.__random_state = random_state
        self.__max_depth =  max_depth
        self.__max_features = max_features
        self.__min_samples_leaf = min_samples_leaf
        self.__min_samples_split = min_samples_split  
        self.__class_weight = class_weight
        self.__criterion = criterion
        
        # Populate the list of decision trees
        for i in range(0,n_estimators):
            clf = DecisionTreeClassifier(
                criterion = 'gini',
                random_state = self.__random_state,
                max_depth = self.__max_depth,
                max_features = self.__max_features,
                min_samples_leaf = self.__min_samples_leaf,
                min_samples_split = self.__min_samples_split
                )
            self.__listOfDecisionTrees.append(clf)
    
    def fit(self, X, Y):
        # Fit the trees on X,Y data, sampling randomly data and features for each tree        
        for i in range(0, self.__n_estimators):
            # Add to data (X) the associated labels in Y 
            X['labels_Y'] = Y
            #Sample, with replacement, training examples from X, Y
            X_samples_1 = X.sample(frac=1.0, replace=True)
            # Sample features (taking the transpose, I'll sample and retranspose again the result)
            # Since we're not considering the last column, there's no risk to sample also the label as a feature
            X_samples_2 = X_samples_1.T[0:len(X_samples_1.columns)-1].sample(frac=1.0, replace=False).T
            # Reassociate the labels
            X_samples_2['labels_Y'] = X_samples_1['labels_Y']
            
            # Create an array to save which features will be used (excluding 'labels_Y')
            features = X_samples_2.columns[:len(X_samples_2.columns)-1]
        
            # Save the list of features considered in this tree, will be used when predicting
            self.__listOfFeaturesPerTree.append(features.tolist())
            # Fit the decision tree on the selected samples and features
            self.__listOfDecisionTrees[i].fit(X_samples_2[features], X_samples_2['labels_Y'])

        return self    
    
    def predict(self, X):
        # Predict labels of new input points X
        
        # Array in which there will be labels suggested by each tree, for each "row" of the data.
        # For each row, the majority of the votes will then be taken
        votes = []
        
        # For each tree, predict the labels of each row and save the results
        for i in range(0, self.__n_estimators):
            features = self.__listOfFeaturesPerTree[i]
            votes.append(self.__listOfDecisionTrees[i].predict(X[features]))
        
        # For each row, compute which was the most present label and select it as final result
        final_votes = []
        for i in range (0, len(X)):
            counts = np.bincount([votes[j][i] for j in range(0,10)])
            final_votes.append(np.argmax(counts))
            
        return final_votes
    
    def score(self, X, Y):
        # Given unlabeled data X and corresponding true labels Y,
        #  return a value in [0,1] representing percentage of correctly labeled data
        predictions = self.predict(X)
        matching = [predictions == Y.values]
        totalMatches = np.sum(matching) / len(predictions)
        return totalMatches
    
    # Function used for cross validation, get actual parameters of the Random Forest
    def get_params(self, deep=True):
        return {
            'n_estimators' : self.__n_estimators,
            'random_state' : self.__random_state,
            'max_depth' : self.__max_depth,
            'max_features' : self.__max_features,
            'min_samples_leaf' : self.__min_samples_leaf,
            'min_samples_split' : self.__min_samples_split,  
            'class_weight' : self.__class_weight,
            'criterion' : self.__criterion
        }


# In[ ]:


# Load data from file into a Pandas dataframe
df = pd.read_csv('processed.cleveland.data', header=None)

# Assign names (and types) of the column to the corresponding data
df.columns = ['age','sex','cp', 'trestbps','chol','fbs', 'restecg', 'thalach','exang','oldpeak','slope','ca','thal','num']
df = df.astype({'age': int, 'sex': int, 'cp': int, 'trestbps': int, 'chol': int, 'fbs': int, 'restecg': int, 'thalach':int, 'exang': int, 'slope': int, 'ca':int, 'thal':int})

# Apply one-hot-encoding: for each categorical feature, add the corresponding columns
df = pd.concat([df, pd.get_dummies(df['cp'], prefix='cp')], axis=1)
df = pd.concat([df, pd.get_dummies(df['restecg'], prefix='restecg')], axis=1)
df = pd.concat([df, pd.get_dummies(df['slope'], prefix='slope')], axis=1)
df = pd.concat([df, pd.get_dummies(df['thal'], prefix='thal')], axis=1)

# Remove the "old" columns
df.drop(['cp'],axis=1, inplace=True)
df.drop(['restecg'],axis=1, inplace=True)
df.drop(['slope'],axis=1, inplace=True)
df.drop(['thal'],axis=1, inplace=True)

# Create a new column called presence representing the presence of heart disease 
df['presence'] = df['num'] > 0 
# If we want to consider the problem as Multiclass classification, comment the previous row and uncomment 
#  the next one
#df['presence'] = df['num']

# Select the type of data in the new column
df = df.astype({'presence': int})

# Rearrange columns order
df = df[['age', 'sex', 'trestbps', 'chol', 'fbs', 'thalach', 'exang', 'oldpeak',
       'ca', 'cp_1', 'cp_2', 'cp_3', 'cp_4', 'restecg_0', 'restecg_1',
       'restecg_2', 'slope_1', 'slope_2', 'slope_3', 'thal_3', 'thal_6',
       'thal_7', 'num', 'presence']]

# Function that, given a length of an array and a percentage p, 
# it will return p% of indexes of that array, randomly chosen, sorted
def defineTraining (length, percentage=0.8):
    sample = random.sample(range(0, length), int(percentage*length))
    sample = np.sort(sample)
    return sample

# Function that:
# - sample 80% of the data and select it as training set
# - use training data to do cross validation (k=10)
# - once it has the best parameters, train the model on the training set
# - test results using test set and returns a percentage of right predictions
def applyRandomForest():
    # Pick 80% of data as training set
    pickAsTraining = defineTraining(len(df), 0.8)
    
    # Add a column to our dataframe, default value will be 0
    df['is_train'] = 0
    
    # For each element selected as training, set its value in 'is_train' to 1
    for el in pickAsTraining :
        df.loc[el,'is_train'] = 1
    
    # Split our dataset in test and train, depending on the value in 'is_train' column
    test, train = df[df['is_train']==0], df[df['is_train']==1]
    # Select the features that will be used for training (in order to not use the labels for training)
    features = df.columns[:22] 
    
    # Initialize our classifier
    clf = MyRandomForest()
    
    # Cross validation - BEGIN 
    # {
    
    # Create arrays in which I'll save the possible values of parameters to be used in RF  
    n_estimators = [800, 1000, 1200]
    max_depth = [60, 100, 140]
    max_depth.append(None)
    max_features = [2, 5, 11]
    min_samples_leaf = [7, 10, 13]
    min_samples_split = [2, 4, 6]
    
    # Create the 'random grid' containing the (selected) possible parameters for RF
    random_grid = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'max_features': max_features,
        'min_samples_leaf': min_samples_leaf,
        'min_samples_split': min_samples_split
    }
    
    # Define an object representing the CV method, already implemented, called on our classifier
    grid_search = GridSearchCV(estimator = clf, param_grid = random_grid, cv = 10, n_jobs = -1, verbose = 1)
    
    # Call the CV model
    grid_search.fit(train[features], train['presence'])
        
    # }    
    # Cross Validation - END
    
    # Redefine our classifier with the found parameters
    clf = MyRandomForest(
        random_state = 0,
        n_estimators = grid_search.best_params_["n_estimators"],
        max_depth = grid_search.best_params_["max_depth"],
        max_features = grid_search.best_params_["max_features"],
        min_samples_leaf = grid_search.best_params_["min_samples_leaf"],
        min_samples_split = grid_search.best_params_["min_samples_split"]
    )
    
    clf.fit(train[features], train['presence'])
    predictions = clf.predict(test[features])
    matching = [predictions == test['presence'].values]
    totalMatches = np.sum(matching) / len(predictions)
    
    return totalMatches


# In[ ]:


# Suppress a warning
pd.options.mode.chained_assignment = None

# Call K times the function "applyRandomForest()"
K = 20
results = np.zeros(K)

for i in range(0,K):
    #if (i%(int)(K/10) == 0):
    #    print ((int)(i/K*100), "%")
    results[i] = applyRandomForest()

#print(results)    
print(np.average(results))
print(np.std(results))

