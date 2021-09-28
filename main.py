import pandas as pd
import numpy as np


# TODO: Implement DTL algorithm and sub procedures
# def DTL(examples, attributes, default):
#     if (examples.len() == 0):
#         return default
#     elif all_class_same():
#         return classification
#     else:
#         best = choose_best_attribute(attributes, examples)
#         tree = Node(best)
#         for val in best:
#             sub_tree = DTL(examples, attributes-best, mode(examples))
#             tree.add_branch(val, sub_tree)
#         return tree
#
# def choose_best_attribute(attributes, examples):
#     max_info_gain = 0
#     best_attribute = None
#     for a in attributes:
#         curr_gain = info_gain(a)
#         if curr_gain > max_info_gain:
#             max_info_gain = curr_gain
#             best_attribute = a
#     return best_attribute
#
#
# def info_gain(attribute):
#     return goal_entropy(examples) - aggregate_entropy(attribute)


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


class Attribute():
    """Class to house details about a particular attribute as the decision tree."""
    def __init__(self, name, outputs):
        self.name = name
        self.outputs = outputs

        # Find all unique types of attribute
        self.types = np.unique(self.outputs)

        # Create a dictionary of outcomes per type, defaulted to 0
        self.type_count = {a_type: [0, 0, 0] for a_type in self.types}
        print(f'--------------{self.name} types: {self.types}--------------\n')
        for i, a_type in enumerate(self.outputs):
            if self.outputs[i] == a_type:
                if OUTCOMES[i]:
                    self.type_count[a_type][0] += 1 # Positive
                else:
                    self.type_count[a_type][1] += 1 # Negative
                self.type_count[a_type][2] += 1 # Total
        print(f'{self.name} frequencies [p, k, v]: {self.type_count}\n')


def goal_entropy():
    """TODO: Calculate the entropy relative to the set of examples. This currently calculates the entropy of the ENTIRE
    data set, not relative to a subset of the data."""
    counts = np.bincount(OUTCOMES)
    prob_p = counts[1] / EXAMPLE_COUNT
    prob_n = 1 - prob_p

    # Note: A H(goal) of 1 means that there are equal # of positive and negative examples
    return np.nan_to_num(-prob_p * np.log2(prob_p) - prob_n * np.log2(prob_n))


def aggregate_entropy(attribute):
    """Calculate the entropy values of each type of an attribute and return aggregate entropy."""
    weighted_sum = 0
    for k in attribute.types:
        print(f'({attribute.name}, {k}):')
        vk = attribute.type_count[k][2]

        prob_p = attribute.type_count[k][0] / vk
        prob_n = attribute.type_count[k][1] / vk
        with np.errstate(divide='ignore', invalid='ignore'): # TODO: Can we safely ignore these warnings?
            entropy = np.nan_to_num(-prob_p * np.log2(prob_p) - prob_n * np.log2(prob_n))
        print(f'Probability (+, -): ({prob_p:.2f}, {prob_n: .2f})\nEntropy: {entropy:.2f}\n')

        # 'weight' the sum of the type entropies.
        weighted_sum += (vk / EXAMPLE_COUNT) * entropy
    print(f'* ({attribute.name}) Aggregate entropy: {weighted_sum}\n')
    return weighted_sum


if __name__ == '__main__':
    # Read in examples from formatted .csv file.
    data = pd.read_csv('data/figure_18-3.csv', sep=',', header=None).to_numpy()

    # Constants related to data read in.
    ATTRIBUTE_COUNT = data.shape[1] - 1
    EXAMPLE_COUNT = data.shape[0] - 1
    ATTRIBUTE_NAMES = np.array(data[0, 1:], dtype=str)
    EXAMPLES = np.array(data[1:, 1:ATTRIBUTE_COUNT], dtype=str)
    OUTCOMES = np.array(data[1:, ATTRIBUTE_COUNT], dtype=int)

    # Goal entropy of ENTIRE data set
    print(f'--------------Goal entropy of ENTIRE data set: {goal_entropy():.2f}--------------\n\n')

    # Store attribute and its outputs (column) as an object in a dictionary.
    attributes = {}
    for i in range(0, EXAMPLES.shape[1]):
        attributes[ATTRIBUTE_NAMES[i]] = Attribute(ATTRIBUTE_NAMES[i], EXAMPLES[:, i].tolist())

        # Perform sample calculations of entropy on scanned examples
        aggregate_entropy(attributes[ATTRIBUTE_NAMES[i]])

    # Example tree node for textbook example
    root = TreeNode("Patrons")
    hungry = TreeNode("Hungry")
    root.children = {"None": False, "Some": True, "Full": hungry}
    typ = TreeNode("Type")
    hungry.children = {1: typ, 0: False}
    fri_sat = TreeNode("Fri/Sat")
    typ.children = {"French": True, "Italian": False, "Thai": fri_sat, "Burger": True}
    fri_sat.children = {0: False, 1: True}

    print(f'--------------Example Tree from Textbook--------------\n\n')
    root.print_tree()
    