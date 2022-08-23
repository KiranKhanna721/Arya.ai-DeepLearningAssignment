import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score,accuracy_score,confusion_matrix

df_train = pd.read_csv('training_set.csv')
df_test = pd.read_csv('test_set.csv')
df_train = df_train.drop(["Unnamed: 0","X2","X12","X25","X26","X27","X28","X29","X30","X31","X32","X33","X34","X35","X36","X37","X38","X39","X40",
                         "X41","X42","X43","X44","X45","X46","X47","X48","X49","X50","X51","X4","X14","X22","X54"],axis=1)
X = df_train.drop("Y",axis=1)
Y = df_train["Y"]
X_train,X_val,y_train,y_val = train_test_split(X,Y,train_size=0.8)
dr=  RandomForestClassifier()
randforest_params = [{'max_depth': list(range(10, 15))}]
clf1 = GridSearchCV(dr, randforest_params, cv = 10, scoring='accuracy')
clf1.fit(X_train, y_train)
print(clf1.best_params_) 
print(clf1.best_score_)
pre_val1 = clf1.predict(X_val)
acc = accuracy_score(y_val,pre_val1)
print("Accuracy Score of Validation Data",acc)
f1 = f1_score(y_val,pre_val1)
print("f1 Score",f1)
conf = confusion_matrix(y_val,pre_val1)
print("Confusion_matrix",conf)
df_test = df_test.drop(["Unnamed: 0","X2","X12","X25","X26","X27","X28","X29","X30","X31","X32","X33","X34","X35","X36","X37","X38","X39","X40",
                         "X41","X42","X43","X44","X45","X46","X47","X48","X49","X50","X51","X4","X14","X22","X54"],axis=1)
pre_test1 = clf1.predict(df_test)
df = pd.DataFrame(pre_test1)
df.to_csv('test_output.csv')