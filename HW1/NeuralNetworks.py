#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 14:52:06 2018

@author: mcampili-warren
"""
#Code taken from CS7641 HW1 by Tian Mi in GitHub
#and adapted to my hw.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

print("RUNNING NeuralNetworks.py NOW")


#################################################
#Letter Recognition dataset

data = pd.read_csv('letter-recognition.csv')
X = data.iloc[:,1:17]
y = data.iloc[:,1]  #Second Column with the letter is our y
features = list(X.columns.values)
#**************************************************************************************************
#Neural network classifier learning curve

list1=[]
list2=[]
print("##########################################")
#print("Doing Neural Network with sgd and logistic" )
start_time = datetime.now()
for i in range(1,95):
    #hidden layers is equal to one layer with the mean of entry and exit, 17 attributes plus one exit, mean 9

    clf = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(9,), random_state=1, activation='logistic')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=1-i/100)
    clf.fit(X_train, y_train)
    train_predict = clf.predict(X_train)
    test_predict = clf.predict(X_test)
    list1.append(accuracy_score(y_train, train_predict))
    list2.append(accuracy_score(y_test, test_predict))
    
print('Duration: {}'.format(datetime.now() - start_time))    
plt.plot(range(len(list2)),list2, color='blue')
plt.plot(range(len(list1)),list1, color='orange')
plt.title('Neural Network Classifier Learning Curve, solver: sgd, activation=logistic')
plt.xlabel('Training in Orange, testing in Blue')
plt.ylabel('Accuracy')
plt.show()

list1=[]
list2=[]
print("##########################################")

#print("Doing Neural Network with sgd and identity" )
start_time = datetime.now()

for i in range(1,95):
   
    clf = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(9,), random_state=1, activation='identity')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=1-i/100)
    clf.fit(X_train, y_train)
    train_predict = clf.predict(X_train)
    test_predict = clf.predict(X_test)
    list1.append(accuracy_score(y_train, train_predict))
    list2.append(accuracy_score(y_test, test_predict))
print('Duration: {}'.format(datetime.now() - start_time))    

plt.plot(range(len(list2)),list2, color='blue')
plt.plot(range(len(list1)),list1, color='orange')
plt.title('Neural Network Classifier Learning Curve, solver: sgd, activation=identity')
plt.xlabel('Training in Orange, testing in Blue')
plt.ylabel('Accuracy')

plt.show()


list1=[]
list2=[]
print("##########################################")

#print("Doing Neural Network with lbfgs and logistic" )
start_time = datetime.now()
for i in range(1,95):
    
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(9,), random_state=1, activation='logistic')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=1-i/100)
    clf.fit(X_train, y_train)
    train_predict = clf.predict(X_train)
    test_predict = clf.predict(X_test)
    list1.append(accuracy_score(y_train, train_predict))
    list2.append(accuracy_score(y_test, test_predict))
print('Duration: {}'.format(datetime.now() - start_time))    

plt.plot(range(len(list2)),list2, color='blue')
plt.plot(range(len(list1)),list1, color='orange')
plt.title('Neural Network Classifier Learning Curve, solver: lbfgs, activation=logistic')
plt.xlabel('Training in Orange, testing in Blue')
plt.ylabel('Accuracy')

plt.show()

list1=[]
list2=[]
print("##########################################")

#print("Doing Neural Network with lbfgs and identity" )
start_time = datetime.now()

for i in range(1,95):

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(9,), random_state=1, activation='identity')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=1-i/100)
    clf.fit(X_train, y_train)
    train_predict = clf.predict(X_train)
    test_predict = clf.predict(X_test)
    list1.append(accuracy_score(y_train, train_predict))
    list2.append(accuracy_score(y_test, test_predict))
print('Duration: {}'.format(datetime.now() - start_time))    

plt.plot(range(len(list2)),list2, color='blue')
plt.plot(range(len(list1)),list1, color='orange')
plt.title('Neural Network Classifier Learning Curve, solver: lbfgs, activation=identity')
plt.xlabel('Training in Orange, testing in Blue')
plt.ylabel('Accuracy')

plt.show()



list1=[]
list2=[]
print("##########################################")

#print("Doing Neural Network with adam and logistic" )
start_time = datetime.now()
for i in range(1,95):
    
    clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(9,), random_state=1, activation='logistic')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=1-i/100)
    clf.fit(X_train, y_train)
    train_predict = clf.predict(X_train)
    test_predict = clf.predict(X_test)
    list1.append(accuracy_score(y_train, train_predict))
    list2.append(accuracy_score(y_test, test_predict))
print('Duration: {}'.format(datetime.now() - start_time))    

plt.plot(range(len(list2)),list2, color='blue')
plt.plot(range(len(list1)),list1, color='orange')
plt.title('Neural Network Classifier Learning Curve, solver: adam, activation=logistic')
plt.xlabel('Training in Orange, testing in Blue')
plt.ylabel('Accuracy')

plt.show()

list1=[]
list2=[]
print("##########################################")

#print("Doing Neural Network with adam and identity" )
start_time = datetime.now()

for i in range(1,95):

    clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(9,), random_state=1, activation='identity')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=1-i/100)
    clf.fit(X_train, y_train)
    train_predict = clf.predict(X_train)
    test_predict = clf.predict(X_test)
    list1.append(accuracy_score(y_train, train_predict))
    list2.append(accuracy_score(y_test, test_predict))
print('Duration: {}'.format(datetime.now() - start_time))    

plt.plot(range(len(list2)),list2, color='blue')
plt.plot(range(len(list1)),list1, color='orange')
plt.title('Neural Network Classifier Learning Curve, solver: adam, activation=identity')
plt.xlabel('Training in Orange, testing in Blue')
plt.ylabel('Accuracy')

plt.show()


#***************************************
#--------Student Performance DATASET
#***************************************


#########################################
#learning curve function from sklearn tutorial

#from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
#from sklearn.model_selection import ShuffleSplit
#from sklearn.ensemble import GradientBoostingClassifier

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):

    """
    Generate a simple plot of the test and training learning curve.
    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.
    title : string
        Title for the chart.
    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.
    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.
    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.
        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.
    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
        """
    

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

###############################################
#Student Performance data set
data = pd.read_csv('Student-Performance-math.csv')
Xx = data.iloc[:,:21]
yy = data.iloc[:,21]
features2 = list(Xx.columns.values)

#Neural network classifier
#hidden layer set to one with 11 neurons (mean of 22 attributes plus one)
print("Doing Neural Network with lbfgs and identity" )
start_time = datetime.now()
clf = MLPClassifier(hidden_layer_sizes=(11,), random_state=1, solver="lbfgs", activation='identity')
plot_learning_curve(clf, "MLP with hidden layers as (11), solver lbfg, activation identity", X, y, ylim=[0,1])   

print("Doing Neural Network with lbfgs and logistic" )
start_time = datetime.now()
clf = MLPClassifier(hidden_layer_sizes=(11,), random_state=1, solver="lbfgs", activation='logistic')
plot_learning_curve(clf, "MLP with hidden layers as (11), solver lbfgs, activation logistic", X, y, ylim=[0,1])   

print("Doing Neural Network with sgd and identity" )
start_time = datetime.now()
clf = MLPClassifier(hidden_layer_sizes=(11,), random_state=1, solver="sgd", activation='identity')
plot_learning_curve(clf, "MLP with hidden layers as (11), solver sgd, activation identity", X, y, ylim=[0,1])   

print("Doing Neural Network with sgd and logistic" )
start_time = datetime.now()
clf = MLPClassifier(hidden_layer_sizes=(11,), random_state=1, solver="sgd", activation='logistic')
plot_learning_curve(clf, "MLP with hidden layers as (11), solver sgd, activation logistic", X, y, ylim=[0,1])   

print("Doing Neural Network with adam and identity" )
start_time = datetime.now()
clf = MLPClassifier(hidden_layer_sizes=(11,), random_state=1, solver="adam", activation='identity')
plot_learning_curve(clf, "MLP with hidden layers as (11), solver adam, activation identity", X, y, ylim=[0,1])  

print("Doing Neural Network with adam and logistic" )
start_time = datetime.now()
clf = MLPClassifier(hidden_layer_sizes=(11,), random_state=1, solver="adam", activation='logistic')
plot_learning_curve(clf, "MLP with hidden layers as (11),solver adam, activation logistic", X, y, ylim=[0,1])   
