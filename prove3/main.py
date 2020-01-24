from statistics import mode

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, train_data, train_target):
        self.train_data = train_data
        self.train_target = train_target

    def predict(self, test_data):
        targets = []
        for test in test_data:
            dist = self.get_distances(test)
            targets.append(self.NN(dist))
        return targets

    def get_distances(self, test):
        distances = []
        for train in self.train_data:
            dist = self.compute_distance(test, train)
            distances.append(dist)
        sorted_indexes = np.argsort(distances)
        return sorted_indexes[:self.k]

    @staticmethod
    def compute_distance(test, train):
        return np.sqrt(np.sum((test - train) ** 2))

    def NN(self, distances):
        target_NN = []
        for d in distances:
            target_NN.append(self.train_target[d])
        return mode(target_NN)


iris = datasets.load_iris()

iris_train, iris_test, target_train, target_test = \
    train_test_split(iris.data, iris.target, test_size=0.30, train_size=0.70, shuffle=True)

classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(iris_train, target_train)
predictions = classifier.predict(iris_test)

print("Predicted:", predictions)
print("Actual:   ", target_test)

count = 0
for idx, p in enumerate(predictions):
    if p == target_test[idx]:
        count += 1

print("Score: ", count/predictions.size)

classifier = KNN(3)
classifier.fit(iris_train, target_train)
target_predicted = classifier.predict(iris_test)

print('Predicted:[', ' '.join(str(p) for p in target_predicted), ']')
# print("Predicted:", target_predicted)
print("Actual:   ", target_test)
count = 0
for idx, p in enumerate(predictions):
    if p == target_test[idx]:
        count += 1

print("Score: ", count/predictions.size)