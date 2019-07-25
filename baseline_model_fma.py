#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import os
import ast

import sklearn as skl
import sklearn.utils, sklearn.preprocessing, sklearn.decomposition, sklearn.svm
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
import numpy as np


# ### Load data

# In[7]:


filepath = '/home/priya/Documents/stanford/cs221/final_project/audio_files/tracks.csv'
tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])
tracks.head()


# In[8]:


keep_cols = [('set', 'split'),
('set', 'subset'),('track', 'genre_top'), ('track', 'genres')]

tracks = tracks[keep_cols]
tracks.head()


# ### Look at the MFCC Features

# In[9]:


filepath = 'features.csv'
features = pd.read_csv(filepath, index_col=0,header=[0, 1, 2], skip_blank_lines=True )


# In[10]:


mfcc = features['mfcc']
mfcc.head()


# ### PCA Analysis and Visualizations

# #### 2 Genres

# In[11]:


small = tracks['set', 'subset'] <= 'small'
genre1 = tracks['track', 'genre_top'] == 'Instrumental'
genre2 = tracks['track', 'genre_top'] == 'Hip-Hop'

print(small.shape, genre1.shape, genre2.shape)

X = features.loc[small & (genre1 | genre2), 'mfcc']
X = skl.decomposition.PCA(n_components=2).fit_transform(X)

y = tracks.loc[small & (genre1 | genre2), ('track', 'genre_top')]
y = skl.preprocessing.LabelEncoder().fit_transform(y)

plt.scatter(X[:,0], X[:,1], c=y, cmap='RdBu', alpha=0.9)
plt.show()
X.shape, y.shape


# #### Multiple Genres

# In[12]:


small = tracks['set', 'subset'] <= 'small'
genre1 = tracks['track', 'genre_top'] == 'Instrumental'
genre2 = tracks['track', 'genre_top'] == 'Hip-Hop'
genre3 = tracks['track', 'genre_top'] == 'Folk'
genre4 = tracks['track', 'genre_top'] == 'Electronic'

print(small.shape, genre1.shape, genre2.shape)

X = features.loc[small & (genre1 | genre2| genre3| genre4), 'mfcc']
X = skl.decomposition.PCA(n_components=2).fit_transform(X)

y = tracks.loc[small & (genre1 | genre2| genre3| genre4), ('track', 'genre_top')]
y = skl.preprocessing.LabelEncoder().fit_transform(y)

plt.scatter(X[:,0], X[:,1], c=y, cmap='RdBu', alpha=0.9)
plt.show()
X.shape, y.shape


# ### Break into train, validation and test

# In[14]:


small = tracks['set', 'subset'] == 'small'

train = tracks['set', 'split'] == 'training'
val = tracks['set', 'split'] == 'validation'
test = tracks['set', 'split'] == 'test'

y_train = tracks.loc[small & train, ('track', 'genre_top')]
y_val = tracks.loc[small & val, ('track', 'genre_top')]
y_test = tracks.loc[small & test, ('track', 'genre_top')]

X_train_mfcc = features.loc[small & train, 'mfcc']
X_val_mfcc = features.loc[small & val, 'mfcc']
X_test_mfcc = features.loc[small & test, 'mfcc']


# X_train = pd.concat([X_train_mfcc, X_train_tonnetz], axis=1)
# X_val = pd.concat([X_val_mfcc, X_val_tonnetz], axis=1)

X_train = X_train_mfcc
X_val = X_val_mfcc

print(X_train_mfcc.shape)

print('{} training examples, {} testing examples'.format(y_train.size, y_val.size))
print('{} features, {} classes'.format(X_train.shape[1], np.unique(y_train).size))


# ### Standardize Features and Encode Labels

# In[15]:


#Shuffle training features
X_train, y_train = skl.utils.shuffle(X_train, y_train, random_state=42)

# Standardize features by removing the mean and scaling to unit variance.
scaler = skl.preprocessing.StandardScaler(copy=False)
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test_mfcc)


# In[16]:


## Label encode y - data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_val = le.fit_transform(y_val)
y_test = le.fit_transform(y_test)
le.classes_


# In[17]:


# X_train[:1]


# In[18]:


y_train[:5]


# ### Build models

# In[19]:


from sklearn.metrics import f1_score
from time import time


# In[25]:


def train_classifier(clf, X_train, y_train):
    ''' Fits a classifier to the training data. '''
    
    # Start the clock, train the classifier, then stop the clock
    start = time()
    clf.fit(X_train, y_train)
    end = time()
    
    # Print the results
    print("Trained model in {:.4f} seconds".format(end - start))

    
def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''
    
    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features)
    end = time()
    
    # Print and return results
    print("Made predictions in {:.4f} seconds.".format(end - start))
    return f1_score(target, y_pred, average='micro', pos_label = 1)


def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Train and predict using a classifer based on F1 score. '''
    
    # Indicate the classifier and the training set size
    print("Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train)))
    
    # Train the classifier
    train_classifier(clf, X_train, y_train)
    
    # Print the results of prediction for both training and testing
    print("F1 score for training set: {:.4f}.".format(predict_labels(clf, X_train, y_train)))
    print("F1 score for val set: {:.4f}.".format(predict_labels(clf, X_val, y_val)))
    print("F1 score for test set: {:.4f}.".format(predict_labels(clf, X_test, y_test)))


# In[26]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
#from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# TODO: Initialize the three models
clf_A = DecisionTreeClassifier(random_state=10, max_depth =4)
clf_B = SVC()
clf_C = LogisticRegression()
clf_D = RandomForestClassifier(random_state=10, max_depth=30, n_estimators=300, min_samples_leaf=6, min_impurity_decrease=0.0002,
                     class_weight='balanced')

for clf in [clf_A, clf_B, clf_C, clf_D]:
    print("\n{}: \n".format(clf.__class__.__name__))
    train_predict(clf, X_train, y_train, X_val, y_val)


# In[27]:


from sklearn.metrics import classification_report

y_true = y_test
y_pred = clf_B.predict(X_test)
labels = [0,1,2,3,4,5,6,7]
target_names = le.inverse_transform(labels)

print(y_true.shape, y_pred.shape)
print(classification_report(y_true, y_pred, target_names=target_names))


# In[28]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_true, y_pred))


# In[29]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_true, y_pred))


# In[ ]:




