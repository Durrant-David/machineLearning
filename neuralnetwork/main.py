from sklearn.model_selection import train_test_split
import pandas as pd
import random
import math


class Node(object):
    def __init__(self):
        self.name = None
        self.h = []
        self.a = []
        self.w = []
        self.e = []


class NeuralNetwork:
    def __init__(self, hidden=2, learning_rate=0.1):
        self.n = learning_rate
        self.inputs = []
        self.outputs = []
        self.hidden = []
        self.i_count = 0
        self.o_count = 0
        self.h_count = 0

        # setup hidden nodes
        for i in range(hidden):
            self.hidden.append(Node())
            self.h_count += 1
        node = Node()
        node.name = "bias"
        node.h.append(-1)
        self.hidden.append(node)
        self.h_count += 1

    def fit(self, train, target):
        self.train = train
        self.target = target
        self.setup_nodes()
        self.inputs, self.hidden, self.outputs = self.load_data(train, self.inputs, self.hidden, self.outputs)
        for i in range(len(self.hidden)):
            self.hidden = self.calc_values(i, train, self.inputs, self.hidden)
        for i in range(len(self.outputs)):
            self.outputs = self.calc_values(i, train, self.hidden, self.outputs)

    def predict(self, test):
        self.p_inputs = self.inputs
        self.p_output = self.outputs
        self.p_hidden = self.hidden
        self.clear_data()

        self.p_inputs, self.p_hidden, self.p_output = self.load_data(test, self.p_inputs, self.p_hidden, self.p_output)
        for i in range(len(self.hidden)):
            self.p_hidden = self.calc_values(i, test, self.p_inputs, self.p_hidden)
        for i in range(len(self.outputs)):
            self.p_output = self.calc_values(i, test, self.p_hidden, self.p_output)
        return self.highest_value()

    def highest_value(self):
        results = []
        comp_outputs = []
        for output in self.outputs:
            comp_outputs.append(output.h)
        df = pd.DataFrame(comp_outputs)
        for i in range(len(comp_outputs[0])):
            results.append(df[:][i].idxmax())
        return results

    def load_data(self, data, inputs, hidden, output):
        for row_index, row in data.iterrows():
            for column in range(self.i_count-1):
                inputs[column].h.append(row[column])
            for i in range(self.h_count):
                hidden[i].h.append(None)
                hidden[i].a.append(None)
            for i in range(self.o_count):
                output[i].h.append(None)
                output[i].a.append(None)
        return inputs, hidden, output

    def clear_data(self):
        for node in self.p_inputs:
            if node.name != "bias":
                node.a = []
                node.h = []
        for node in self.p_hidden:
            if node.name != "bias":
                node.a = []
                node.h = []
        for node in self.p_output:
            if node.name != "bias":
                node.a = []
                node.h = []

    def calc_values(self, index, train, inputs, outputs):
        for i in range(len(train)):
            h = 0
            for node in inputs:
                if node.name != "bias":
                    h += node.h[i] * node.w[index]
                else:
                    h += node.h[0] * node.w[index]
            outputs[index].a[i] = self.sigmoid(h)
            outputs[index].h[i] = h
        return outputs

    def setup_nodes(self):
        # setup inputs nodes
        for column in self.train.columns:
            node = Node()
            node.name = column
            self.inputs.append(node)
            self.i_count += 1
        node = Node()
        node.name = "bias"
        node.h.append(-1)
        self.inputs.append(node)
        self.i_count += 1

        # setup outputs nodes
        for value in self.target.unique():
            node = Node()
            node.name = value
            self.outputs.append(node)
            self.o_count += 1

        # add weights
        for node in self.inputs:
            for i in range(self.h_count):
                node.w.append(random.uniform(0, 1))
        for node in self.hidden:
            for i in range(self.o_count):
                node.w.append(random.uniform(0, 1))

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))


car_header = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
car_cleanup_data = {"buying": {"vhigh": 4, "high": 3, "med": 2, "low": 1},
                    "maint": {"vhigh": 4, "high": 3, "med": 2, "low": 1},
                    "doors": {"5more": 5},
                    "persons": {"more": 6},
                    "lug_boot": {"small": 1, "med": 2, "big": 3},
                    "safety": {"low": 1, "med": 2, "high": 3}}

car = pd.read_csv("data/car.data", names=car_header)

obj_car = car.select_dtypes(include=['object']).copy()
obj_car.replace(car_cleanup_data, inplace=True)
obj_car.doors = pd.to_numeric(obj_car.doors)
obj_car.persons = pd.to_numeric(obj_car.persons)

x = obj_car.drop(["class"], axis=1)
y = obj_car["class"]

iris_train, iris_test, target_train, target_test = \
    train_test_split(x, y, test_size=0.30, train_size=0.70, shuffle=True)

classifier = NeuralNetwork(hidden=2)
classifier.fit(iris_train, target_train)
predictions = classifier.predict(iris_test)
print(predictions)
print(target_test)
