import numpy as np
import pandas as pd
from tpot import TPOTClassifier
from sklearn.cross_validation import train_test_split

#load data
telescope = pd.read_csv('Data.csv')

#clean data
telescope_shuffle = telescope.iloc[np.random.permutation(len(telescope))]
tele = telescope_shuffle.reset_index(drop=True)

#Store 2 classes
tele['Class'] = tele['Class'].map({'g':0,'h':1})
tele_class = tele['Class'].values

#Split dataset for training and testing
X_train,X_test = y_train,y_test = train_test_split(tele.index,stratify = tele_class,train_size =0.75,test_size = 0.25)

#Genetic Programming to find best ML model and hyperparameters
tpot = TPOTClassifier(generations = 5,verbosity=2)
tpot.fit(tele.drop('Class',axis=1).loc[X_train].values,
    tele.loc[y_train,'Class'].values)

#Score accuracy
tpot.score(tele.drop('Class',axis = 1).loc[X_test].values,
        tele.loc[X_test,'Class'].values)

#Export the generated code
tpot.export('pipeline.py')
