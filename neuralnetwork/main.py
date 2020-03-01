from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nodes.InputNode import InputNode as iNode
from nodes.HiddenNode import HiddenNode as hNode
from nodes.OutputNode import OutputNode as oNode
from sklearn.datasets import load_iris

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


class NeuralNetwork:
    def __init__(self, hidden=2, learning_rate=0.1):

        # for graph
        self.outputs_graph = [[]]

        self.n = learning_rate
        self.inputs = iNode()
        self.outputs = oNode()
        self.hidden = hNode()

        self.hidden.build_nodes(hidden)

    def fit(self, train, target):
        self.train = train
        self.target = target
        if isinstance(self.train, np.ndarray):
            self.inputs.build_nodes(range(self.train.shape[1]))
            self.outputs.build_nodes(sorted(np.unique(self.target, axis=0)))
        else:
            self.inputs.build_nodes(self.train.columns)
            self.outputs.build_nodes(sorted(self.target.unique()))

        # add weights
        self.inputs.add_weights(self.hidden.node_count)
        self.hidden.add_weights(self.outputs.node_count)

        # for testing
        # self.inputs.nodes[0].w[0] = 0.3
        # self.inputs.nodes[0].w[1] = 0.2
        # self.inputs.nodes[1].w[0] = 0.4
        # self.inputs.nodes[1].w[1] = -0.1
        # self.inputs.nodes[2].w[0] = 0.2
        # self.inputs.nodes[2].w[1] = -0.1
        # end testing

        # for testing
        # self.hidden.nodes[0].w[0] = 0.2
        # self.hidden.nodes[0].w[1] = -0.2
        # self.hidden.nodes[1].w[0] = -0.1
        # self.hidden.nodes[1].w[1] = 0.4
        # self.hidden.nodes[2].w[0] = 0.3
        # self.hidden.nodes[2].w[1] = 0.1
        # end testing

        self.inputs.add_inputs(train)
        self.hidden.init_arrays(train.shape[0])
        self.outputs.init_arrays(train.shape[0], self.target)
        for node_i in range(self.outputs.node_count):
            self.outputs_graph.append([])

        i = 0
        while i < 10000:
            i += 1
            for row_i in range(self.outputs.row_count):
                self.hidden.calc_values(self.inputs.nodes, row_i)
                self.outputs.calc_values(self.hidden.nodes, row_i)

                for node_i in range(self.outputs.node_count):
                    self.outputs.calc_error(node_i, row_i)
                for node_i in range(self.hidden.node_count):
                    self.hidden.update_weights(node_i, self.outputs.nodes, self.n, row_i)
                    self.hidden.calc_error(node_i, self.outputs, row_i)
                for node_i in range(self.hidden.node_count - 1):
                    self.inputs.update_weights(node_i, self.hidden.nodes[node_i].e, self.n, row_i)

            for node_i in range(self.outputs.node_count):
                self.outputs_graph[node_i].append(self.outputs.nodes[node_i].e)

    def predict(self, test):
        self.p_inputs = iNode()
        self.p_outputs = oNode()
        self.p_hidden = hNode()

        # copy layout of fit
        self.p_inputs.copy_layout(self.inputs)
        self.p_hidden.copy_layout(self.hidden)
        self.p_outputs.copy_layout(self.outputs)

        self.p_inputs.add_inputs(test)
        self.p_hidden.init_arrays(test.shape[0])
        if isinstance(self.train, np.ndarray):
            self.p_outputs.init_arrays(test.shape[0], sorted(np.unique(self.target, axis=0)))
        else:
            self.p_outputs.init_arrays(test.shape[0], sorted(self.target.unique()))

        rows = self.p_outputs.row_count
        for row_i in range(rows):
            self.p_hidden.calc_values(self.inputs.nodes, row_i)
            self.p_outputs.calc_values(self.p_hidden.nodes, row_i)

        return self.highest_value()

    def highest_value(self):
        results = []
        comp_outputs = []
        for output in self.p_outputs.nodes:
            comp_outputs.append(output.a)
        df = pd.DataFrame(comp_outputs)
        df[df < 0.5] = 0
        df[df > 0.4] = 1
        for i in range(len(comp_outputs[0])):
            results.append(df[:][i].idxmax())
        return df


car_header = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
car_cleanup_data = {"buying": {"vhigh": 4, "high": 3, "med": 2, "low": 1},
                    "maint": {"vhigh": 4, "high": 3, "med": 2, "low": 1},
                    "doors": {"5more": 5},
                    "persons": {"more": 6},
                    "lug_boot": {"small": 1, "med": 2, "big": 3},
                    "safety": {"low": 1, "med": 2, "high": 3},
                    "class": {"unacc": 0, "acc": 1, "good": 2, "vgood": 3}}

car = pd.read_csv("data/car.data", names=car_header)

obj_car = car.select_dtypes(include=['object']).copy()
obj_car.replace(car_cleanup_data, inplace=True)
obj_car.doors = pd.to_numeric(obj_car.doors)
obj_car.persons = pd.to_numeric(obj_car.persons)

x = obj_car.drop(["class"], axis=1)
y = obj_car["class"]

iris_train, iris_test, target_train, target_test = \
    train_test_split(x, y, test_size=0.30, train_size=0.70, shuffle=True)

classifier = NeuralNetwork(hidden=2, learning_rate=0.01)
# FIRST DATA SET
# classifier.fit(iris_train, target_train)
# predictions = classifier.predict(iris_test)
# print(predictions)
# print(target_test)
# for output in classifier.outputs_graph:
#     plt.plot(output)
# plt.show()

simple = [[1.2, -0.2, 0], [0.8, 0.1, 1]]
header = ["x1", "x2", "Y"]
df = pd.DataFrame(simple)
df.columns = header

# SECOND DATA SET
# x = df.drop(["Y"], axis=1)
# y = df["Y"]
# classifier.fit(x, y)
# predictions = classifier.predict(x)

# print(predictions)
# print(df["Y"])
# for output in classifier.outputs_graph:
#     plt.plot(output)
# plt.show()

iris = load_iris()
x = iris.data
y = iris.target
iris_train, iris_test, target_train, target_test = \
    train_test_split(x, y, test_size=0.30, train_size=0.70, shuffle=True)

classifier.fit(iris_train, target_train)
predictions = classifier.predict(iris_test)
print(predictions)
print(target_test)
for output in classifier.outputs_graph:
    plt.plot(output)
plt.show()