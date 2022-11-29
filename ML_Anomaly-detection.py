from sklearn import preprocessing
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score,precision_score, recall_score,confusion_matrix,ConfusionMatrixDisplay

import pandas as pd

features = ["record ID", "duration_", "src_bytes", "dst_bytes"]

df = pd.read_csv('conn_attack.csv', names=features, header=None)
df_labels = pd.read_csv('conn_attack_anomaly_labels.csv', names=["record ID", "malicious?"], header=None)

dataset = df.to_numpy()
dataset = np.delete(dataset, 0, 1)

labels = df_labels.to_numpy()
labels = np.delete(labels,0,1).flatten()

duration_train, duration_test, src_train, src_test, dest_train, dest_test, labels_train, labels_test = \
    train_test_split(dataset[:, 0], dataset[:, 1], dataset[:, 2],labels , test_size=0.2)

x_train = np.stack((duration_train, src_train, dest_train), axis=1)
x_test = np.stack((duration_test, src_test, dest_test), axis=1)

scaler = preprocessing.StandardScaler().fit(x_train)
x_train_transformed = scaler.transform(x_train)

scaler = preprocessing.StandardScaler().fit(x_test)
x_test_transformed = scaler.transform(x_test)

# clustering1 = DBSCAN(eps=0.0001, min_samples=3).fit(x_train_transformed)
#
iso_forest = IsolationForest().fit(x_train)

y_pred1 = iso_forest.fit_predict(x_test)
# print(y_pred1)
#
y_pred1=list(map(lambda x: 1 if x==-1 else 0,y_pred1))
#
print("accuracy1 score: {0:.2f}%".format(accuracy_score(labels_test, y_pred1) * 100))
print("precision1 score: {0:.2f}%".format(precision_score(labels_test, y_pred1) * 100))
print("recall1 score: {0:.2f}%".format(recall_score(labels_test, y_pred1) * 100))
print(confusion_matrix(labels_test, y_pred1))
ConfusionMatrixDisplay.from_predictions(labels_test, y_pred1, labels=[1,0])

plt.show()