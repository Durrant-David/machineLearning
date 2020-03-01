import math


class Node(object):
    def __init__(self, value=None, name=None):
        if value is None:
            value = []
        self.name = name
        self.values = value
        self.h = []
        self.a = []
        self.w = []
        self.e = []

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))