#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 13:09:20 2018

@author: mcampili-warren
"""

#!/usr/bin/python3
#Code taken from CS7641 HW1 by Tian Mi in GitHub
#and adapted to my hw.


import numpy as np
import pandas as pd
import graphviz
import matplotlib.pyplot as plt
import sklearn.tree as tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import os

print("RUNNING DecisionTreeBoosting.py NOW")


#################################################
#Letter Recognition dataset
print("Using LETTER RECOGNITION DATAFILE now")

data = pd.read_csv('letter-recognition.csv')
X = data.iloc[:,1:17]
y = data.iloc[:,1]  #Second Column with the letter is our y
features = list(X.columns.values)


#decision tree learning curve of function of split gini
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.20)
list1=[]
list2=[]
for depth in range(3,40):
    clf = DecisionTreeClassifier(random_state=1, criterion='gini', max_depth=depth)
    clf = clf.fit(X_train, y_train)
    test_predict = clf.predict(X_test)
    list1.append(accuracy_score(y_test, test_predict))
        
plt.plot(range(len(list1)),list1, color='green')
plt.xlabel('tree depth')
plt.ylabel('accuracy')
plt.show()


#decision tree learning curve of tree depth 
list1=[]
list2=[]
for i in range(1,95):
    clf = DecisionTreeClassifier(random_state=1, criterion='gini', max_depth=8)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=1-i/100)
    test_size=1-i/100

    clf = clf.fit(X_train, y_train)
    
    train_predict = clf.predict(X_train)
    test_predict = clf.predict(X_test)
    
    list1.append(accuracy_score(y_train, train_predict))
    list2.append(accuracy_score(y_test, test_predict))
    
plt.plot(range(len(list2)),list2, color='blue')
plt.plot(range(len(list1)),list1, color='orange')
plt.xlabel('Max_depth:8-Training in orange, Testing in blue both with GINI')
plt.ylabel('Accuracy')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.20)

clf = DecisionTreeClassifier(random_state=1, criterion='gini', max_depth=8)
clf = clf.fit(X_train, y_train)
test_predict = clf.predict(X_test)
print("The prediction accuracy of decision tree with max_depth 8 is " + str(accuracy_score(y_test, test_predict)))

#visualization of decision tree
dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=features,  
                         class_names=list(map(str, set(y))),  
                         filled=True, rounded=True,  
                         special_characters=True)
graph1 = graphviz.Source(dot_data)
graph1.view()  #will create a "Source.gv.pdf" with the graph of the tree


os.rename('Source.gv.pdf', 'LETTERS-REC-gini-DEPTH-8.gv.pdf')


#decision tree learning curve of tree depth
list1=[]
list2=[]
for i in range(1,95):
    clf = DecisionTreeClassifier(random_state=1, criterion='gini', max_depth=9)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=1-i/100)
    test_size=1-i/100

    clf = clf.fit(X_train, y_train)
    
    train_predict = clf.predict(X_train)
    test_predict = clf.predict(X_test)
    
    list1.append(accuracy_score(y_train, train_predict))
    list2.append(accuracy_score(y_test, test_predict))
    
plt.plot(range(len(list2)),list2, color='blue')
plt.plot(range(len(list1)),list1, color='orange')
plt.xlabel('Max_depth:9-Training in orange, Testing in blue both with GINI')
plt.ylabel('Accuracy')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.20)

clf = DecisionTreeClassifier(random_state=1, criterion='gini', max_depth=9)
clf = clf.fit(X_train, y_train)
test_predict = clf.predict(X_test)
print("The prediction accuracy of decision tree with max_depth 9 is " + str(accuracy_score(y_test, test_predict)))

#visualization of decision tree
dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=features,  
                         class_names=list(map(str, set(y))),  
                         filled=True, rounded=True,  
                         special_characters=True)
graph1 = graphviz.Source(dot_data)
graph1.view()  #will create a "Source.gv.pdf" with the graph of the tree


os.rename('Source.gv.pdf', 'LETTERS-REC-gini-DEPTH-9.gv.pdf')


#decision tree learning curve of tree depth
list1=[]
list2=[]
for i in range(1,95):
    clf = DecisionTreeClassifier(random_state=1, criterion='gini', max_depth=10)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=1-i/100)
    test_size=1-i/100

    clf = clf.fit(X_train, y_train)
    
    train_predict = clf.predict(X_train)
    test_predict = clf.predict(X_test)
    
    list1.append(accuracy_score(y_train, train_predict))
    list2.append(accuracy_score(y_test, test_predict))
    
plt.plot(range(len(list2)),list2, color='blue')
plt.plot(range(len(list1)),list1, color='orange')
plt.xlabel('Max_depth:10-Training in orange, Testing in blue both with GINI')
plt.ylabel('Accuracy')
plt.show()

#Optimal solution
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.20)

clf = DecisionTreeClassifier(random_state=1, criterion='gini', max_depth=10)
clf = clf.fit(X_train, y_train)
test_predict = clf.predict(X_test)
print("The prediction accuracy of decision tree with depth_math=10 is " + str(accuracy_score(y_test, test_predict)))

#visualization of decision tree
dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=features,  
                         class_names=list(map(str, set(y))),  
                         filled=True, rounded=True,  
                         special_characters=True)
graph1 = graphviz.Source(dot_data)
graph1.view()  #will create a "Source.gv.pdf" with the graph of the tree


os.rename('Source.gv.pdf', 'LETTERS-REC-gini-DEPTH-10.gv.pdf')


#########################################
#  BOOSTING
#########################################
#Boosted DT classifier
list1=[]
list2=[]
start_time = datetime.now()
print('#############################################')
print('AdaBoostClassifier with default n_estimators=50:')
for i in range(1,95):
    clf = AdaBoostClassifier()  #default n_estimators = 50, original code set to 100
    #In case of perfect fit, the learning procedure is stopped early.
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=1-i/100)
    clf.fit(X_train, y_train)
    train_predict = clf.predict(X_train)
    test_predict = clf.predict(X_test)
    list1.append(accuracy_score(y_train, train_predict))
    list2.append(accuracy_score(y_test, test_predict))
print('Duration: {}'.format(datetime.now() - start_time))    
plt.plot(range(len(list2)),list2, color='blue')
plt.plot(range(len(list1)),list1,color='orange')
plt.xlabel('Boosting with n_estimators=50, Training in orange, Testing in blue')
plt.ylabel('Accuracy')

plt.show()

list1=[]
list2=[]
print('#############################################')
print('AdaBoostClassifier with default n_estimators=100:')
start_time = datetime.now()

for i in range(1,95):
    clf = AdaBoostClassifier(n_estimators=500)  
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=1-i/100)
    clf.fit(X_train, y_train)
    train_predict = clf.predict(X_train)
    test_predict = clf.predict(X_test)
    list1.append(accuracy_score(y_train, train_predict))
    list2.append(accuracy_score(y_test, test_predict))
print('Duration: {}'.format(datetime.now() - start_time))    
plt.plot(range(len(list2)),list2, color='blue')
plt.plot(range(len(list1)),list1,color='orange')
plt.xlabel('Boosting with n_estimators=100, Training in orange, Testing in blue')
plt.ylabel('Accuracy')

plt.show()

list1=[]
list2=[]
print('#############################################')
print('AdaBoostClassifier with default n_estimators=500:')
print('WARNING - IT TAKES 11 MINUTES TO COMPLETE')
start_time = datetime.now()

for i in range(1,95):
    clf = AdaBoostClassifier(n_estimators=500)  
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=1-i/100)
    clf.fit(X_train, y_train)
    train_predict = clf.predict(X_train)
    test_predict = clf.predict(X_test)
    list1.append(accuracy_score(y_train, train_predict))
    list2.append(accuracy_score(y_test, test_predict))
print('Duration: {}'.format(datetime.now() - start_time))    
plt.plot(range(len(list2)),list2, color='blue')
plt.plot(range(len(list1)),list1,color='orange')
plt.xlabel('Boosting with n_estimators=500, Training in orange, Testing in blue')
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
print('##############################################################')
print("Using STUDENT PERFORMANCE DATAFILE now")

#Decision Tree classifier

#decision tree learning curve of tree depth 3
clf3 = DecisionTreeClassifier(random_state=1, criterion='gini', max_depth=3)
plot_learning_curve(clf3, "Decision Tree with max depth 3", Xx, yy, ylim=[0,1])

#decision tree learning curve of tree depth 4
clf4 = DecisionTreeClassifier(random_state=1, criterion='gini', max_depth=4)
plot_learning_curve(clf4, "Decision Tree with max depth 4", Xx, yy, ylim=[0,1])

#decision tree learning curve of tree depth 5
clf5 = DecisionTreeClassifier(random_state=1, criterion='gini', max_depth=5)
plot_learning_curve(clf5, "Decision Tree with max depth 5", Xx, yy, ylim=[0,1])

clf3 = clf3.fit(Xx, yy)
dot_data2 = tree.export_graphviz(clf3, out_file=None, 
                         feature_names=features2,  
                         class_names=list(map(str, set(yy))),  
                         filled=True, rounded=True,  
                         special_characters=True)

graphT = graphviz.Source(dot_data2)
graphT.view()  #will create a "Source.gv.pdf" with the graph of the tree


os.rename('Source.gv.pdf', 'STUDENT-tree.gv.pdf')

####################
#Boosting
###################
#Boosted DT classifier

print('DOING BOOSTING NOW')
print('AdaBoostingClassifier with estimators=50')
clf = AdaBoostClassifier(n_estimators=50)
plot_learning_curve(clf, "Adaboost with n_estimators 50", Xx, yy, ylim=[0,1])

print('AdaBoostingClassifier with estimators=100')
clf = AdaBoostClassifier(n_estimators=100)
plot_learning_curve(clf, "Adaboost with n_estimators 100", Xx, yy, ylim=[0,1])

#print('AdaBoostingClassifier with estimators=100')
#clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.001, max_depth=3, random_state=1, max_leaf_nodes=5)
#plot_learning_curve(clf, "Gradient Boosting with n_estimators 1000", Xx, yy, ylim=[0,1],train_sizes=np.linspace(.1, 1.0))


