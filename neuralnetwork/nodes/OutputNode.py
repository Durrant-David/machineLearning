from nodes.Node import Node


class OutputNode(object):
    def __init__(self):
        self.nodes = []
        self.node_count = 0
        self.row_count = 0

    def build_nodes(self, output_values):
        for value in output_values:
            self.nodes.append(Node(name=value))
            self.node_count += 1

    def init_arrays(self, rows, targets):
        self.row_count = rows
        for i in range(self.node_count):
            self.nodes[i].h = [None] * rows
            self.nodes[i].a = [None] * rows
        for target in targets:
            self.nodes[0].values.append(target)
        for node in self.nodes:
            node.values = self.nodes[0].values

    def calc_values(self, inputs, row):
        for i_output in range(self.node_count):
            h = 0
            for node in inputs:
                if node.name != "bias":
                    h += node.a[row] * node.w[i_output]
                else:
                    h += node.values * node.w[i_output]
            self.nodes[i_output].a[row] = Node.sigmoid(h)
            self.nodes[i_output].h[row] = h


    def copy_layout(self, oNode):
        self.node_count = oNode.node_count
        for i in range(self.node_count):
            node = Node(name=oNode.nodes[i].name)
            node.w = oNode.nodes[i].w
            self.nodes.append(node)

    def calc_error(self, index, row):
        a = self.nodes[index].a[row]
        # t = self.nodes[index].values[row]
        if self.nodes[index].values[row] == index:
            t = 1
        else:
            t = 0
        # e = (a) * (1 - a) * (a - t)
        self.nodes[index].e = a * (1 - a) * (a - t)
