import random
from nodes.Node import Node
import numpy as np


class InputNode(object):
    def __init__(self):
        self.nodes = []
        self.node_count = 0

    def build_nodes(self, columns):
        for column in columns:
            self.nodes.append(Node(name=column))
            self.node_count += 1
        node = Node()
        node.name = "bias"
        node.values.append(-1)
        self.nodes.append(node)
        self.node_count += 1

    def add_weights(self, num_next_layer):
        for node in self.nodes:
            # -1 for bias node
            for i in range(num_next_layer - 1):
                node.w.append(random.uniform(0, 1))

    def add_inputs(self, data):
        if isinstance(data, np.ndarray):
            for row in data:
                for column in range(self.node_count - 1):
                    self.nodes[column].values.append(row[column])
        else:
            for row_index, row in data.iterrows():
                for column in range(self.node_count - 1):
                    self.nodes[column].values.append(row[column])

    def copy_layout(self, iNode):
        self.node_count = iNode.node_count
        for i in range(self.node_count):
            node = Node(iNode.nodes[i].values, iNode.nodes[i].name)
            node.w = iNode.nodes[i].w
            self.nodes.append(node)

    def update_weights(self, index, error, learning_rate, row):
        # w = w - n * e[k] * a[i]
        for i, weight in enumerate(self.nodes[index].w):
            if self.nodes[index].name == "bias":
                self.nodes[index].w[i] = weight - \
                                         learning_rate * \
                                         error * \
                                         self.nodes[index].values
            else:
                self.nodes[index].w[i] = weight - \
                                         learning_rate * \
                                         error * \
                                         self.nodes[index].values[row]