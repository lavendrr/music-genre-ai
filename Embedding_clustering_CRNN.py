#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
from os.path import isfile
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Bidirectional, LSTM, Dropout, Activation, GRU
from keras.layers import Conv2D, concatenate, MaxPooling2D, Flatten, Embedding, Lambda


from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras import backend as K
from keras.utils import np_utils
from keras.optimizers import Adam, RMSprop

from keras import regularizers


import librosa
import librosa.display
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Load pretrained model and test data

# In[2]:


dict_genres = {'Electronic':0, 'Experimental':1, 'Folk':2, 'Hip-Hop':3, 
               'Instrumental':4,'International':5, 'Pop' :6, 'Rock': 7  }


# In[3]:


from keras.models import load_model

weights_path = 'models/crnn/weights.best.h5'
model = load_model(weights_path)


# In[4]:


npzfile = np.load('test_arr.npz')
print(npzfile.files)
X_test = npzfile['arr_0']
y_test = npzfile['arr_1']
print(X_test.shape, y_test.shape)


# In[5]:


y_test -= 1
print(np.amin(y_test), np.amax(y_test), np.mean(y_test))


# In[6]:


X_test_raw = librosa.core.db_to_power(X_test, ref=1.0)
X_test = np.log(X_test_raw)
# X_test = np.expand_dims(X_test, axis = -1)
print(X_test.shape)


# ### Extract embeddings from the concat layer of the model

# In[7]:


layer_name = 'dense1'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(X_test)


# In[8]:


print(intermediate_output.shape)


# In[9]:


# print(intermediate_output[:2])


# ### Cluster using K means

# In[10]:


from sklearn.cluster import KMeans, SpectralClustering


# In[11]:


kmeans = KMeans(n_clusters=8, init='random', verbose =1).fit(intermediate_output)


# In[12]:


labels = kmeans.labels_
labels.shape


# In[13]:


print(labels)


# In[ ]:





# ### Evaluate the output from K-means 

# #### Adjusted Rand Index

# In[14]:


from sklearn.metrics import adjusted_rand_score


# In[15]:


adjusted_rand_score(y_test, labels)


# #### Try different number of clusters

# In[16]:


from sklearn.metrics import silhouette_score

for cluster in range(2,10):
    kmeans = KMeans(n_clusters=cluster, init='random', verbose =0).fit(intermediate_output)
    labels = kmeans.labels_
    score = silhouette_score(intermediate_output, labels, metric='euclidean')
    print("Cluster number and Score is: ", cluster, score)


# ### Visualizations

# In[17]:


### 8 clusters
from sklearn.metrics import confusion_matrix
import seaborn as sns
cluster = 8
kmeans = KMeans(n_clusters=cluster, init='random', verbose =0).fit(intermediate_output)
labels = kmeans.labels_

mat = confusion_matrix(y_test, labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=dict_genres.keys())
plt.xlabel('true label')
plt.ylabel('predicted label');


# In[19]:


### 4 clusters
from sklearn.metrics import confusion_matrix
import seaborn as sns
cluster = 4
kmeans = KMeans(n_clusters=cluster, init='random', verbose =0).fit(intermediate_output)
labels = kmeans.labels_

mat = confusion_matrix(y_test, labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=dict_genres.keys())
plt.xlabel('true label')
plt.ylabel('predicted label');


# In[ ]:




