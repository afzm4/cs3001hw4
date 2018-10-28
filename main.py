# -*- coding: utf-8 -*-
"""
Andrew Floyd
CS3001 - HW4 (NaiveBayesian)
October 28th, 2018
"""

import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

data = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

data["Home_Away"]=np.where(data["Home/Away"]=="Home",0,1)
data["AP_25"]=np.where(data["AP25"]=="In",0,1)
data["Media_m"]=np.where(data["Media"]=="1-NBC",0,np.where(data["Media"]=="2-ESPN",1,np.where(data["Media"]=="3-FOX",2,np.where(data["Media"]=="4-ABC",3,np.where(data["Media"]=="5-CBS",4,5)))))
data["Label_m"]=np.where(data["Label"]=="Win",0,1)

test["Home_Away"]=np.where(test["Home/Away"]=="Home",0,1)
test["AP_25"]=np.where(test["AP25"]=="In",0,1)
test["Media_m"]=np.where(test["Media"]=="1-NBC",0,np.where(test["Media"]=="2-ESPN",1,np.where(test["Media"]=="3-FOX",2,np.where(test["Media"]=="4-ABC",3,np.where(test["Media"]=="5-CBS",4,5)))))
test["Label_m"]=np.where(test["Label"]=="Win",0,1)

data_cleaned = data[["ID","Home_Away","AP_25","Media_m","Label_m"]]
test_cleaned = test[["ID","Home_Away","AP_25","Media_m","Label_m"]]

gnb = GaussianNB()

used_features =[
        "Home_Away",
        "AP_25",
        "Media_m"]

gnb.fit(
        data_cleaned[used_features].values,
        data_cleaned["Label_m"])
y_pred = gnb.predict(test_cleaned[used_features])

y_true = np.array([0,1,0,0,0,0,0,0,0,1,0,1])

accu = accuracy_score(y_true, y_pred)
prec, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred, average="binary")

print("Accuracy: ", accu)
print("Precision: ", prec)
print("Recall: ", recall)
print("F1: ", fscore)
print("")
print("Where 0 = Win and 1 = Lose:")
print(y_pred)