
# coding: utf-8

# In[1]:


# Import useful stuff from libraries etc

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

import pandas as pd
import numpy as np
import random 

np.random.seed(0)


# In[2]:


# Load data from file into a dataframe
df = pd.read_csv('processed.cleveland.data', header=None)

# Assign the names of the column to the corresponding data
df.columns = ['age','sex','cp', 'trestbps','chol','fbs', 'restecg', 'thalach','exang','oldpeak','slope','ca','thal','num']
df = df.astype({'age': int, 'sex': int, 'cp': int, 'trestbps': int, 'chol': int, 'fbs': int, 'restecg': int, 'thalach':int, 'exang': int, 'slope': int, 'ca':int, 'thal':int})


# <p>
#     <br> <b>Description of features: </b>
#     <br> Age = age in years 
#     <br> Sex = sex (0 = female; 1 = male)
#     <br> cp = chest pain type
#         <br> &nbsp; -- Value 1: typical angina
#         <br> &nbsp; -- Value 2: atypical angina
#         <br> &nbsp; -- Value 3: non-anginal pain
#         <br> &nbsp; -- Value 4: asymptomatic
#     <br> trestbps = resting blood pressure (in mm Hg on admission to the hospital)
#     <br> chol = serum cholestoral in mg/dl
#     <br> fbs = (fasting blood sugar > 120 mg/dl)  (1 = true; 0 = false)
#     <br> restecg = resting electrocardiographic results
#         <br> &nbsp; -- Value 0: normal
#         <br> &nbsp; -- Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
#         <br> &nbsp; -- Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria
#     <br> thalach = maximum heart rate achieved
#     <br> exang = exercise induced angina (1 = yes; 0 = no)
#     <br> oldpeak = ST depression induced by exercise relative to rest
#     <br> slope = the slope of the peak exercise ST segment
#         <br> &nbsp;-- Value 1: upsloping
#         <br> &nbsp;-- Value 2: flat
#         <br> &nbsp;-- Value 3: downsloping
#     <br> ca = number of major vessels (0-3) colored by flourosopy
#     <br> thal = ?
#         <br> &nbsp;-- 3 = normal
#         <br> &nbsp;-- 6 = fixed defect
#         <br> &nbsp;-- 7 = reversable defect
#     <br> num = (0 = absence; 1,2,3,4 = presence)
#     
#     <hr>
# </p>

# <p> 
#     Some features have categorical data...in order to pass from categorical data to non-categorical data <br>
#     I will apply <i>One Hot Encoding</i> <br><br>
#     Features before: <br>
#     <b>age</b>, <b>sex</b>, cp, <b>trestbps</b>, <b>chol</b>, <b>fbs</b>, restecg, <b>thalach</b>, <b>exang</b>, <b>oldpeak</b>, slope, <b>ca</b>, thal <br>
#     (in bold the non categorical ones) <br> <br>
#     
#     cp, restecg, slope and thal are categorical <br>
#     - <b>cp</b> has 4 different values, so it will become: cp1, cp2, cp3, cp4 <br>
#     - <b>restecg</b> has 3 different values, so it will become: restecg0, restecg1, restecg2 <br>
#     - <b>slope</b> has 3 different values, so it will become: slope1, slope2, slope3 <br>
#     - <b>thal</b> has 3 different values, so it will become: thal3, thal6, thal7 <br>
# </p>

# In[3]:


# For each feature, add the corresponding columns
df = pd.concat([df, pd.get_dummies(df['cp'], prefix='cp')], axis=1)
df = pd.concat([df, pd.get_dummies(df['restecg'], prefix='restecg')], axis=1)
df = pd.concat([df, pd.get_dummies(df['slope'], prefix='slope')], axis=1)
df = pd.concat([df, pd.get_dummies(df['thal'], prefix='thal')], axis=1)

# Remove the "old" column
df.drop(['cp'],axis=1, inplace=True)
df.drop(['restecg'],axis=1, inplace=True)
df.drop(['slope'],axis=1, inplace=True)
df.drop(['thal'],axis=1, inplace=True)

# Create a new column called presence representing the presence of heart disease 
df['presence'] = df['num'] > 0
#df['presence'] = df['num'] 
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
    clf = RandomForestClassifier(n_jobs = -1, random_state=0)
    
    # Cross validation - BEGIN 
    # {
    
    # Create arrays in which I'll save the possible values of parameters to be used in RF   
    n_estimators = [800, 1000, 1200]
    max_depth = [60, 100, 140]
    max_depth.append(None)
    max_features = [2, 5, 11]
    min_samples_leaf = [7, 10, 13]
    min_samples_split = [2, 4, 6]
    
    # Create the random grid containing the possible parameters for RF
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
    
    # Redefine our classifier with the found parameters
    clf = RandomForestClassifier(
        n_jobs = -1,
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


# In[4]:


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

