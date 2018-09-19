#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 19:52:04 2018

@author: mcampili-warren

"""
#Code taken from CS7641 HW1 by Tian Mi in GitHub
#and adapted to my hw.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from datetime import datetime

print("RUNNING SVMandKNN.py NOW")

#################################################
#Letter Recognition dataset
print("Using LETTER RECOGNITION DATAFILE now")

data = pd.read_csv('letter-recognition.csv')
X = data.iloc[:,1:17]
y = data.iloc[:,1]  #Second Column with the letter is our y
features = list(X.columns.values)
print('############ SVM #################')

#SVM classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.20)
for kernel in ('linear', 'rbf'):
    clf = svm.SVC(kernel=kernel, gamma='auto')
    clf = clf.fit(X_train, y_train)
    test_predict = clf.predict(X_test)
    print("Accuracy with "+str(kernel)+" is "+ str(accuracy_score(y_test, test_predict)))


#SVM learning curve with RBF kernel
list1=[]
list2=[]
start_time = datetime.now()

for i in range(1,95):
    clf = svm.SVC(kernel="rbf", gamma='auto')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=1-i/100)
    clf.fit(X_train, y_train)
    train_predict = clf.predict(X_train)
    test_predict = clf.predict(X_test)
    list1.append(accuracy_score(y_train, train_predict))
    list2.append(accuracy_score(y_test, test_predict))
print('Duration: {}'.format(datetime.now() - start_time))    

plt.ylim(ymin=0,ymax=1.1)
plt.plot(range(len(list2)),list2, color= 'blue')
plt.plot(range(len(list1)),list1, color='orange')
plt.title('SVM Learning Curve with RBF kernel')
plt.xlabel('Training in Orange, testing in Blue')
plt.ylabel('Accuracy')
plt.show()

#SVM learning curve with linear kernel
list1=[]
list2=[]
start_time = datetime.now()

for i in range(1,95):
    clf = svm.SVC(kernel="linear", gamma='auto')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=1-i/100)
    clf.fit(X_train, y_train)
    train_predict = clf.predict(X_train)
    test_predict = clf.predict(X_test)
    list1.append(accuracy_score(y_train, train_predict))
    list2.append(accuracy_score(y_test, test_predict))
print('Duration: {}'.format(datetime.now() - start_time))    

plt.ylim(ymin=0,ymax=1.1)
plt.plot(range(len(list2)),list2, color= 'blue')
plt.plot(range(len(list1)),list1, color='orange')
plt.title('SVM Learning Curve with linear kernel')
plt.xlabel('Training in Orange, testing in Blue')
plt.ylabel('Accuracy')
plt.show()

print('############ KNN #################')

#KNN classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.20)
KNN_list=[]
list2=[]

print("KNN learning curve, cv=5 folds, k=1..50")
for K in range(1,50):
    clf = KNeighborsClassifier(K, weights="distance")
    scores = cross_val_score(clf, X_train, y_train, cv=5)
    clf = clf.fit(X_train, y_train)
    test_predict = clf.predict(X_test)
    KNN_list.append(accuracy_score(y_test, test_predict))
    list2.append(sum(scores)/len(scores))
plt.plot(range(len(KNN_list)),KNN_list, color='green')
plt.plot(range(len(list2)),list2, color='red')
plt.title('KNN Classifier')
plt.xlabel('Cross Validation in red, KNN Accuracy in green ')
plt.ylabel('Accuracy')
plt.show()

print("KNN learning curve, cv=10 folds, k=1..50")
KNN_list=[]
list2=[]

for K in range(1,50):
    clf = KNeighborsClassifier(K, weights="distance")
    scores = cross_val_score(clf, X_train, y_train, cv=10)
    clf = clf.fit(X_train, y_train)
    test_predict = clf.predict(X_test)
    KNN_list.append(accuracy_score(y_test, test_predict))
    list2.append(sum(scores)/len(scores))
plt.plot(range(len(KNN_list)),KNN_list, color='green')
plt.plot(range(len(list2)),list2, color='red')
plt.title('KNN Classifier')
plt.xlabel('Cross Validation in red, KNN Accuracy in green ')
plt.ylabel('Accuracy')
plt.show()

#choose 13 as the optimal K
clf = KNeighborsClassifier(13, weights="distance")
score = cross_val_score(clf, X_train, y_train, cv=10)
clf = clf.fit(X_train, y_train)
test_predict = clf.predict(X_test)
print("The prediction accuracy of KNN classifier with k=13 " + str(accuracy_score(y_test, test_predict)))
print("cross validation score "+str(sum(score)/len(score)))
#learning curve of KNN classifier with K=13
list1=[]
list2=[]
start_time = datetime.now()

for i in range(1,95):
    clf = KNeighborsClassifier(13, weights="distance")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=1-i/100)
    clf.fit(X_train, y_train)
    train_predict = clf.predict(X_train)
    test_predict = clf.predict(X_test)
    list1.append(accuracy_score(y_train, train_predict))
    list2.append(accuracy_score(y_test, test_predict))
print('Duration: {}'.format(datetime.now() - start_time))    

plt.plot(range(len(list2)),list2, color='blue')
plt.plot(range(len(list1)),list1, color='orange')
plt.xlabel('training in orange, testing in glue  ')
plt.ylabel('Accuracy')
plt.show()
#***************************************
#--------Student Performance DATASET
#***************************************


#########################################
#learning curve function from sklearn tutorial


from sklearn.model_selection import learning_curve

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

print('##############################################################')
print("Using STUDENT PERFORMANCE DATAFILE now")

#SVM classifier
from sklearn.model_selection import GridSearchCV
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100],
                     'C': [0.01, 0.1, 1, 10, 100, 1000]} ]
clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=5, scoring='accuracy')
clf.fit(Xx, yy)
print(clf.best_params_)

tuned_parameters = [{'kernel': ['sigmoid'], 'gamma': [1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100],
                     'C': [0.01, 0.1, 1, 10, 100, 1000]} ]
clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=5, scoring='accuracy')
clf.fit(Xx, yy)
print(clf.best_params_)

tuned_parameters = [{'kernel': ['poly'], 'gamma': [1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100],
                     'C': [0.01, 0.1, 1, 10, 100, 1000]} ]
clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=5, scoring='accuracy')
clf.fit(Xx, yy)
print(clf.best_params_)

clf = svm.SVC(C=0.01, kernel="rbf", gamma=0.0001)
plot_learning_curve(clf, "SVM with RBF, gamma=0.0001", Xx, yy, ylim=[0,1.2],train_sizes=np.linspace(.4, 1.0))


clf = svm.SVC(C=1, kernel="sigmoid", gamma=0.001)
plot_learning_curve(clf, "SVM with Sigmoid kernel, gamma=0.0001", Xx, yy, ylim=[0,1.2],train_sizes=np.linspace(.4, 1.0))

clf = svm.SVC(C=0.01, kernel="poly", gamma=0.0001)
plot_learning_curve(clf, "SVM with poly, gamma=0.0001", Xx, yy, ylim=[0,1.2],train_sizes=np.linspace(.4, 1.0))


#KNN classifier
clf = KNeighborsClassifier(1, weights="distance", p=2)
plot_learning_curve(clf, "K nearest neighbors with K=1", Xx, yy, ylim=[0,1.2])

clf = KNeighborsClassifier(5, weights="distance", p=2)
plot_learning_curve(clf, "K nearest neighbors with K=5", Xx, yy, ylim=[0,1.2])

clf = KNeighborsClassifier(10, weights="distance", p=2)
plot_learning_curve(clf, "K nearest neighbors with K=10", Xx, yy, ylim=[0,1.2])

clf = KNeighborsClassifier(10, weights="uniform", p=2)
plot_learning_curve(clf, "K nearest neighbors with K=10, uniform weights", Xx, yy, ylim=[0,1.2])