# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 19:43:43 2024

@author: Cassandra DeBlois
"""

import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix 
from sklearn import linear_model, tree, svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

plankton = pd.read_csv("Plankton.csv", index_col=False)
fish = pd.read_csv("Fish.csv", index_col=False)

for column in plankton[1:]:
    #Used -1 to 0.001 to include just the 0 values since that is most of them
    plankton[column] = pd.cut(x=plankton[column], bins=[-1, 0.001, 99, 500, 2000,999999], 
                     labels=[1,2,3,4,5])

 
for column in fish[1:]:
    fish[column] = pd.cut(x=fish[column], bins=[-1, 20, 40, 50, 60, 80, 100, 120, 140, 160, 1000], 
                     labels=[1,2,3,4,5,6,7,8,9,10])
    
#All of this below was me testing which machine learning model to use on some preliminary data. All three got perfect accuracy and f1 scores!
    #I left it in here to show the models I tests, linear regression failed and also doesn't make sense here, KNeighbors also failed

# x= plankton
# y = fish["Scup"]
# X_train, X_test, Y_train, Y_test = train_test_split(x, y, train_size=.6)


# tree = tree.DecisionTreeClassifier()
# tree_parameters = {'max_depth':[2, 5, 10], 'min_samples_leaf':[5, 15, 30, 50]}
# gs_tree = GridSearchCV(tree, tree_parameters)
# gs_tree.fit(X_train, Y_train)
# print(gs_tree.best_params_)
# gs_tree.fit(X_train,Y_train)
# tree_pred = gs_tree.predict(X_test)
# print("dtc confusion matirx: " + str(confusion_matrix(Y_test, tree_pred)))
# print("dtc f1 score: " + str(f1_score(Y_test, tree_pred)))
# print("dtc accuracy: " + str(accuracy_score(Y_test,tree_pred)))

# svm = svm.SVC()
# svm_parameters = {'kernel':['rbf', 'linear', 'poly'], 'C':[1, 10, 15], 'degree':[1,3,5,10]}
# gs_svm = GridSearchCV(svm, svm_parameters)
# gs_svm.fit(X_train, Y_train)
# print(gs_svm.best_params_)
# svm_pred = gs_svm.predict(X_test)
# print("svm confusion matirx: " + str(confusion_matrix(Y_test, svm_pred)))
# print("svm f1 score: " + str(f1_score(Y_test, svm_pred)))
# print("svm accuracy: " + str(accuracy_score(Y_test,svm_pred)))

# gnb = GaussianNB()
# gnb_parameters = {}
# gs_gnb = GridSearchCV(gnb, gnb_parameters)
# gs_gnb.fit(X_train, Y_train)
# print(gs_gnb.best_params_)
# gs_gnb.fit(X_train,Y_train)
# gnb_pred = gs_gnb.predict(X_test)
# print("gnb confusion matirx: " + str(confusion_matrix(Y_test, gnb_pred)))
# print("gnb f1 score: " + str(f1_score(Y_test, gnb_pred)))
# print("gnb accuracy: " + str(accuracy_score(Y_test,gnb_pred)))


#Problem species that don't have enough variation for proper modeling
fish = fish.drop(columns=['Butterfish'])
fish = fish.drop(columns=['Cancer crabs'])
fish = fish.drop(columns=['Lobster'])
fish = fish.drop(columns=['Scup'])
fish = fish.drop(columns=['Long finned squid'])
fish = fish.drop(columns=['Winter flounder'])


#This doesn't need to be repeated in the loop
x= plankton

#For each species of fish...
for column in fish[1:]:
    
    print(column) #Fish species
    y = fish[column]
    
    #Randomly split the data into 60% training data and 40% testing data
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, train_size=.6)
    
    #Create a new decision tree classifier
    treeClassifier = tree.DecisionTreeClassifier()
    
    #See what parameters are best for the model to use here
    treeClassifier_parameters = {'max_depth':[2, 5, 10], 'min_samples_leaf':[5, 15, 30, 50]}
    gs_treeClassifier = GridSearchCV(treeClassifier, treeClassifier_parameters)
    
    #Train the model on the training data (and tell us what the best parameters ended up being)
    gs_treeClassifier.fit(X_train, Y_train)
    print(gs_treeClassifier.best_params_)
    
    #Test the model on the testing data and see how it did!
    treeClassifier_pred = gs_treeClassifier.predict(X_test)
    print("dtc confusion matirx: " + str(confusion_matrix(Y_test, treeClassifier_pred)))
    print("dtc f1 score: " + str(f1_score(Y_test, treeClassifier_pred)))
    print("dtc accuracy: " + str(accuracy_score(Y_test,treeClassifier_pred)))

