import random

from nodes.Node import Node


class HiddenNode(object):
    def __init__(self):
        self.nodes = []
        self.node_count = 0
        self.row_count = 0

    def build_nodes(self, num=2):
        for i in range(num):
            self.nodes.append(Node())
            self.node_count += 1

        node = Node(-1, "bias")
        self.nodes.append(node)
        self.node_count += 1

    def add_weights(self, num_next_layer):
        for node in self.nodes:
            for i in range(num_next_layer):
                node.w.append(random.uniform(0, 1))

    def init_arrays(self, rows):
        self.row_count = rows
        for i in range(self.node_count):
            self.nodes[i].h = [None] * rows
            self.nodes[i].a = [None] * rows

    def calc_values(self, inputs, row):
        for i_hidden in range(self.node_count-1):
            h = 0
            for node in inputs:
                if node.name != "bias":
                    h += node.values[row] * node.w[i_hidden]
                else:
                    h += node.values[0] * node.w[i_hidden]
            self.nodes[i_hidden].a[row] = Node.sigmoid(h)
            self.nodes[i_hidden].h[row] = h

    def copy_layout(self, hNode):
        self.node_count = hNode.node_count
        for i in range(self.node_count):
            node = Node(hNode.nodes[i].values, hNode.nodes[i].name)
            node.w = hNode.nodes[i].w
            self.nodes.append(node)

    def update_weights(self, index, error, learning_rate, row):
        # w = w - n * e[k] * a[i]
        for i, weight in enumerate(self.nodes[index].w):
            if self.nodes[index].name == "bias":
                self.nodes[index].w[i] = weight - \
                                         learning_rate * \
                                         error[i].e * \
                                         self.nodes[index].values
            else:
                self.nodes[index].w[i] = weight - \
                                         learning_rate * \
                                         error[i].e * \
                                         self.nodes[index].a[row]

    def calc_error(self, index, n_layer, row):
        # e = (a) * (1 - a) * ((w0) * (e0) + (w1) * (e1))
        if self.nodes[index].name == "bias":
            a = self.nodes[index].values
        else:
            a = self.nodes[index].a[row]
        loop = 0
        for weight, node in zip(self.nodes[index].w, n_layer.nodes):
            loop += weight * node.e
        self.nodes[index].e = a * (1 - a) * loop
