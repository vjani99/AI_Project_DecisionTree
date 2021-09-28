import pandas as pd
import numpy as np


class TreeNode:
    """A data structure for a node in a directed decision tree"""
    def __init__(self, name):
        """Initialize a node with a dictionary of children and it's name"""
        self.children = {}
        self.name = name

    def add_branch(self, value, node):
        """Map a node to a child of boolean value or a subtree root node"""
        self.children[value] = node

    def print_tree(self):
        """Print all nodes of tree (parent with children below)"""
        print("\n" + self.name)
        print(self.children)

        for child in self.children.values():
            if isinstance(child, TreeNode):
                child.print_tree()



if __name__ == '__main__':
    data = pd.read_csv('data/figure_18-3.csv', sep=',').to_numpy()
    ATTRIBUTE_COUNT = data.shape[1] - 1

    # examples = np.array(data[:, 1:ATTRIBUTE_COUNT], dtype=str)
    # classifications = np.array(data[:, ATTRIBUTE_COUNT], dtype=int)

    # print(examples)
    # print(classifications)

    root = TreeNode("Patrons")
    hungry = TreeNode("Hungry")
    root.children = {"None": False, "Some": True, "Full": hungry}
    typ = TreeNode("Type")
    hungry.children = {1: typ, 0: False}
    fri_sat = TreeNode("Fri/Sat")
    typ.children = {"French": True, "Italian": False, "Thai": fri_sat, "Burger": True}
    fri_sat.children = {0: False, 1: True}

    root.print_tree()
