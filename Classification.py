import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

RAMDOM_SEED = 999

class Classification():
    def __init__(self, data):
        self.X_train, self.X_test, self.y_train, self.y_test = data
    
    def classify(self, model):
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        print('Accuracy: {}'.format(accuracy_score(self.y_test, y_pred)))


    def Kmeans(self):
        n_clusters = len(np.unique(self.y_train))
        clf = KMeans(n_clusters = n_clusters, random_state=RAMDOM_SEED)
        clf.fit(self.X_train)
        y_labels_train = clf.labels_
        y_labels_test = clf.predict(self.X_test)
        self.X_train = y_labels_train[:, np.newaxis]
        self.X_test = y_labels_test[:, np.newaxis]

        return self
if __name__ == "__main__":
    Classification(load_digits())\
        .Kmeans()\
        .classify(model=SVC())