import math
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


class Node(object):
    def __init__(self):
        self.child = []
        self.parent = None
        self.attribute = None
        self.data = None


class DecisionTree:
    def fit(self, train_data, train_target):
        self.train_data = train_data
        self.train_target = train_target
        self.attributes = list(range(0, len(train_data[0])))
        self.node = Node()
        self.get_node(self.train_data, dict(enumerate(train_target.flatten(), 1)), self.attributes, self.node)

    def get_node(self, data, targets, attributes, parent_node):
        if all(target == self.train_target[targets[0]] for target in self.train_target[targets]):
            # if all target data is the same then end branch
            parent_node.data = self.train_target[targets[0]]
        elif len(attributes) < 1:
            # if there are no more attributes end branch
            parent_node.data = self.train_target[targets[0]]
        else:
            # separate attributes into branches
            branches, entropies = self.get_branches(attributes, data, targets)

            # get lowest entropy attribute
            parent_node.attribute = entropies.index(min(entropies))
            parent_node.data = branches[parent_node.attribute]

            # remove attribute that have been selected
            del attributes[parent_node.attribute]
            new_data = np.delete(data, np.s_[parent_node.attribute], axis=1)
            for branch in branches[parent_node.attribute]:
                # Recursive function call
                parent_node.child.append(Node())
                self.get_node(new_data, branch, attributes, parent_node.child[len(parent_node.child)-1])

    def get_branches(self, attributes, data, target):
        attribute_indices = []
        branch_entropy = []

        # for each column
        for label in range(len(attributes)):
            column = data[:, label]
            branch_indices = []

            # for each unique value in column
            for branch in set(column):
                indices = []
                for value in range(len(column)):
                    if np.all((value in target) & (column[value] == branch)):
                        # get indices of each value
                        indices.append(value)
                branch_indices.append(indices)

            # add branch indices
            attribute_indices.append(branch_indices)
            # get target value connected to branch_indices
            branch_values = self.get_targets(branch_indices)

            set_entropy = []
            splits = []
            # for each branch in attribute
            for values in branch_values:
                # get occurrence of each value
                value_count = np.bincount(values)
                entropy = []
                # for each unique value in branch
                for value in value_count:
                    # get value count / total values
                    entropy.append(value / len(values))
                # get entropy of branch
                set_entropy.append(self.entropy(entropy))
                splits.append(len(values) / len(column))

            field_entropy = []
            # for each branch entropy
            for entropy, split in zip(set_entropy, splits):
                # values in branch / total values in parent node
                field_entropy.append([split, entropy])
            # get information gain
            branch_entropy.append(self.gain(field_entropy))

        # return the indices connected to each attributes branches and entropy
        return attribute_indices, branch_entropy

    def get_targets(self, branch):
        values = []
        for attribute in branch:
            values.append(target[attribute])
        return values


    # get the entropy of the values in decision
    # values = (value_x_count / total_values) foreach value in category
    def entropy(self, values):
        score = 0
        for v in values:
            if v == 0:
                score -= 0
            else:
                score -= (v * math.log2(v))

        return score

    # get the entropy of the sets from category
    # values = [(category_x_count / total_category_values), set_entropy_score] foreach category
    def gain(self, values):
        score = 0
        for value in values:
            score += (value[0] * value[1])

        return score


# iris = datasets.load_iris()
iris = np.array([[2, 1, 1],
                 [2, 1, 0],
                 [2, 0, 1],
                 [2, 0, 0],
                 [1, 1, 1],
                 [1, 0, 0],
                 [1, 1, 0],
                 [1, 0, 1],
                 [0, 1, 1],
                 [0, 1, 0],
                 [0, 0, 1],
                 [0, 0, 0]])
target = np.array([1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0])
# iris_train, iris_test, target_train, target_test = \
#     train_test_split(iris, target, test_size=0.00, train_size=1.00, shuffle=False)

classifier = DecisionTree()
classifier.fit(iris, target)
print(classifier.node.data)
print(classifier.node.child[0].data, "  -  ", classifier.node.child[0].data[1])
# predictions = classifier.predict(iris_test)
