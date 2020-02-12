import math
import copy
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split


class Node(object):
    def __init__(self):
        self.child = []
        self.parent = None
        self.attribute = None
        self.targets = None
        self.predictions = None


class DecisionTree:
    def fit(self, train_data, train_target):
        self.train_data = train_data
        self.train_target = train_target
        self.attributes = train_data.columns #list(range(0, len(train_data[0])))
        self.node = self.make_tree(self.train_data, self.train_target, self.attributes) #list(range(1, len(self.train_target) + 1)), self.attributes)

    def predict(self, test_data):
        self.test_data = test_data
        self.attributes = test_data.columns  # list(range(0, len(train_data[0])))
        self.check_tree(self.train_data, self.attributes, self.node)  # list(range(1, len(self.train_target) + 1)), self.attributes)

    def check_tree(self, data, features, node):
        for row in data:
            row[node.attribute]

    def make_tree(self, data, targets, features):
        node = Node()
        node.targets = targets
        # for value in targets:
        #     target_values.append(self.train_target[value])
        if all(item == targets[0] for item in targets) or (len(features) == 0):
            return node

        feature_targets = self.get_feature_branches(data, targets, features)
        info_gain = self.get_feature_entropy(feature_targets, targets)
        index_min = np.argmin(info_gain)
        node.attribute = features[index_min]

        rem_data = copy.deepcopy(data)
        rem_data = rem_data.drop([node.attribute], axis=1) #np.delete(rem_data, node.attribute, 1)
        rem_features = rem_data.columns
        for branch in feature_targets[index_min]:
            node.child.append(self.make_tree(rem_data, branch, rem_features))

        return node

    def get_feature_branches(self, data, targets, features):
        all = []
        for feature in features:
            column = data.loc[:, feature]
            feature_targets = []
            for branch in set(column):
                branch_targets = []
                for value in range(len(column)):
                    if np.all(((value + 1) in targets) & (column[value] == branch)):
                        # get indices of each value
                        branch_targets.append(value + 1)
                feature_targets.append(branch_targets)
            all.append(feature_targets)
        return all

    def get_feature_entropy(self, feature_targets, target_values):
        info_gain = []
        for feature in feature_targets:
            feature_prep = []
            for branch in feature:
                target_branch = [self.train_target[i] for i in branch]  # target_values[branch]
                # for item in branch:
                #     target_branch.append(self.train_target[item])
                branch_prep = []
                for count in np.bincount(target_branch):
                    branch_prep.append(count / len(branch))
                feature_prep.append(len(branch) / len(target_values))
                feature_prep.append(self.calc_entropy(branch_prep))
            info_gain.append(self.calc_info_gain(feature_prep))
        return info_gain

    # get the entropy of the values in decision
    # values = (value_x_count / total_values) foreach value in category
    def calc_entropy(self, values):
        score = 0
        for v in values:
            if v == 0:
                score -= 0
            else:
                score -= (v * math.log2(v))

        return score

    # get the entropy of the sets from category
    # values = [(category_x_count / total_category_values), set_entropy_score] foreach category
    def calc_info_gain(self, values):
        score = 0
        i = 0
        while i < len(values):
            score += (values[i] * values[i + 1])
            i += 2
        return score


car_header = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
car_cleanup_data = {"buying": {"vhigh": 4, "high": 3, "med": 2, "low": 1},
                    "maint": {"vhigh": 4, "high": 3, "med": 2, "low": 1},
                    "doors": {"5more": 5},
                    "persons": {"more": 6},
                    "lug_boot": {"small": 1, "med": 2, "big": 3},
                    "safety": {"low": 1, "med": 2, "high": 3},
                    "class": {"vgood": 4, "good": 3, "acc": 2, "unacc": 2}}

car = pd.read_csv("data/car.data", names=car_header)

obj_car = car.select_dtypes(include=['object']).copy()
obj_car.replace(car_cleanup_data, inplace=True)

x = obj_car.drop(["class"], axis=1)
y = obj_car["class"]

iris_train, iris_test, target_train, target_test = \
    train_test_split(x, y, test_size=0.30, train_size=0.70, shuffle=False)

classifier = DecisionTree()
classifier.fit(iris_train, target_train)
classifier.predict(iris_test)
print(classifier.node.targets)

# iris = datasets.load_iris()
# iris = np.array([[2, 1, 1],
#                  [2, 1, 0],
#                  [2, 0, 1],
#                  [2, 0, 0],
#                  [1, 1, 1],
#                  [1, 0, 0],
#                  [1, 1, 0],
#                  [1, 0, 1],
#                  [0, 1, 1],
#                  [0, 1, 0],
#                  [0, 0, 1],
#                  [0, 0, 0]])
# target = np.array([1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0])
# iris_train, iris_test, target_train, target_test = \
#     train_test_split(iris.data, iris.target, test_size=0.30, train_size=0.70, shuffle=False)
#
# classifier = DecisionTree()
# classifier.fit(iris.data, iris.target)
# print(classifier.node.targets)
# predictions = classifier.predict(iris_test)
