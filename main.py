from collections import defaultdict

import pandas as pd
import numpy as np
from sys import maxsize


# TODO: Implement DTL algorithm and sub procedures
def DTL(examples, attributes, default):
    if examples.size == 0:
        # print(f"No more examples left for attributes: {attributes}")
        return default
    elif all_classes_same(examples):
        # Since we know all classes are same, just return classification of first example
        # print(f"All classification are the same for: {attributes}")
        return bool(int(examples[0, -1]))
    elif not attributes:
        # print(f"No attributes left for remaining examples: {examples}")
        return mode(examples)
    else:
        # Update the frequencies of attribute types with new set of examples
        for a in attributes.values():
            a.update_counts(examples[:, a.col_idx].T, get_outcomes(examples))

        best = choose_best_attribute(attributes, examples)
        # print(f"Best attribute: {best.name}")
        tree = TreeNode(best.name)

        # Remove best attribute from dictionary (copy)
        subset_attributes = attributes.copy()
        del subset_attributes[best.name]

        # Divide examples by the types of the best attribute.
        for val in best.types:
            ex_val_subset = get_examples_by_type(examples, val, best.col_idx)
            sub_tree = DTL(ex_val_subset, subset_attributes, mode(examples))
            tree.add_branch(val, sub_tree)
        return tree


def get_examples_by_type(examples, a_type, a_col_idx):
    """Get subset of examples that match this attribute type"""
    return examples[np.where(examples[:, a_col_idx] == a_type)]


def mode(examples):
    outcomes = get_outcomes(examples)    # Extract row of all outcomes
    counts = np.bincount(outcomes)       # Count occurrences of 0/1

    # print(f"counts {counts}")
    if counts.size != 1 and counts[0] == counts[1]:
        print("Tie in outcomes, choosing False")

    # TODO: This ALWAYS chooses the first max calculated for tie-breaking (0)... may want to randomize
    mode = counts.argmax()
    # print(f"Mode {mode}")

    return bool(mode)    # Calculate mode and return as a boolean


def get_outcomes(examples):
    """Return outcomes of examples as a 1D array."""
    return np.array(examples[:, -1].T, np.int64)


def all_classes_same(examples):
    outcomes = get_outcomes(examples)           # Extract row of all outcomes
    return np.all(outcomes == outcomes[0])      # Return whether all occurrences are the same


def choose_best_attribute(att_dict, examples):
    max_info_gain = -maxsize
    best_attribute = None
    outcomes = get_outcomes(examples)
    # print(f'Examples: {examples}')
    # print(f'Outcomes: {outcomes}')
    # print(f'Remaining attributes: {[a.name for a in att_dict.values()]}')
    for a in att_dict.values():
        curr_gain = info_gain(a, examples)
        # print(f"{a.name} info gain: {curr_gain}")
        if curr_gain > max_info_gain:
            max_info_gain = curr_gain
            best_attribute = a
    return best_attribute


def info_gain(attribute, examples):
    goal = goal_entropy(examples)
    aggregate = aggregate_entropy(attribute, examples.shape[0])
    # print(f'Remaining Examples: {examples.shape[0]}')
    # print(f'Attribute {attribute.name}: {attribute.type_count}')
    # print(f'Goal Entropy: {goal}')
    # print(f'Aggregate Entropy: {aggregate}')
    return goal - aggregate


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
        print("\n" + self.name + "?")
        # print(self.children)
        print([f"{key}: Tree({c.name})" if isinstance(c, TreeNode) else f"{key}: {c}" for key, c in self.children.items()])

        for child in self.children.values():
            if isinstance(child, TreeNode):
                child.print_tree()


class Attribute():
    """Class to house details about a particular attribute as the decision tree."""
    def __init__(self, name, outputs, col_idx):
        self.name = name
        self.col_idx = col_idx

        # Find all unique types of attribute
        self.types = np.unique(outputs)
        self.type_count = {}
        for t in self.types:
            self.type_count.setdefault(t, [0, 0, 0])

        # Set initial frequencies
        # self.update_counts(outputs, outcomes)

    def update_counts(self, outputs, outcomes):
        """Given a row of all outputs for this attribute and outcomes for those outputs, store frequencies of
        each output value."""

        # Create a dictionary of outcomes per type, defaulted to 0
        self.types = np.unique(outputs)
        self.type_count.clear()
        # for t in self.types:
            # self.type_count.setdefault(t, [0, 0, 0])
        # self.type_count = defaultdict([0, 0, 0])
        self.type_count = {a_type: [0, 0, 0] for a_type in self.types}
        # print(f'--------------{self.name} types: {self.types}--------------\n')
        for i, a_type in enumerate(outputs):
            # print(f'ith output: {outputs[i]}, a_type: {a_type}')
            if outcomes[i]:
                self.type_count[a_type][0] += 1     # Positive
            else:
                self.type_count[a_type][1] += 1     # Negative
            self.type_count[a_type][2] += 1         # Total
        # for key, val in list(self.type_count.items()):
        #     if val[2] == 0:
        #         del self.type_count[key]
        # print(f'{self.name} frequencies [p, k, v]: {self.type_count}\n')


def goal_entropy(examples):
    """Calculate the entropy relative to the subset of examples/outcomes."""
    outcomes = get_outcomes(examples)  # Extract row of all outcomes
    counts = np.bincount(outcomes)

    # Calculate probability of true/false
    prob_p = counts[1] / examples.shape[0]
    prob_n = 1 - prob_p
    # print(f"Positive prob {prob_p}, Negative prob {prob_n}")

    # Calculate and return goal entropy of this example subset.
    # Note: A H(goal) of 1 means that there are equal # of positive and negative examples
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.nan_to_num(-prob_p * np.log2(prob_p) - prob_n * np.log2(prob_n))


def aggregate_entropy(attribute, example_count):
    """Calculate the entropy values of each type of an attribute and return aggregate entropy."""
    weighted_sum = 0
    for k in attribute.types:
        print(f'({attribute.name}, {k}):')
        vk = attribute.type_count[k][2]

        prob_p = attribute.type_count[k][0] / vk
        prob_n = attribute.type_count[k][1] / vk
        with np.errstate(divide='ignore', invalid='ignore'): # TODO: Can we safely ignore these warnings?
            entropy = np.nan_to_num(-prob_p * np.log2(prob_p) - prob_n * np.log2(prob_n))
        # print(f'Probability (+, -): ({prob_p:.2f}, {prob_n: .2f})\nEntropy: {entropy:.2f}\n')

        # 'weight' the sum of the type entropies.
        weighted_sum += (vk / example_count) * entropy
    # print(f'* ({attribute.name}) Aggregate entropy: {weighted_sum}\n')
    return weighted_sum


if __name__ == '__main__':
    # Read in examples from formatted .csv file.
    data = pd.read_csv('data/simplified-figure_18-3.csv', sep=',', header=None).to_numpy()

    # Constants related to data read in.
    ATTRIBUTE_COUNT = data.shape[1] - 1
    EXAMPLE_COUNT = data.shape[0] - 1
    ATTRIBUTE_NAMES = np.array(data[0, 1:], dtype=str)

    # Read in all attributes, AND the outcome in the last index
    EXAMPLES = np.array(data[1:, 1:ATTRIBUTE_COUNT+1], dtype=str)

    ###### Special cases for testing ######
    # EXAMPLES = np.append(EXAMPLES, [EXAMPLES[0]], axis=0) # Unbalanced data set
    # print(EXAMPLES)
    # EXAMPLES = EXAMPLES[0]    # A single example data set
    ######################################

    EXAMPLES = np.atleast_2d(EXAMPLES)  # Ensure that a 1D array is still treated as 2D in all calculations.

    # mode(EXAMPLES)
    # print(all_classes_same(EXAMPLES))

    # # Goal entropy of ENTIRE data set
    # print(f'--------------Goal entropy of ENTIRE data set: {goal_entropy(EXAMPLES):.2f}--------------\n\n')
    #
    attributes = {}
    for i in range(0, EXAMPLES.shape[1] - 1):
        attributes[ATTRIBUTE_NAMES[i]] = Attribute(ATTRIBUTE_NAMES[i], EXAMPLES[:, i].T, i)
    #
    #     # Perform sample calculations of entropy on scanned examples
    #     aggregate_entropy(attributes[ATTRIBUTE_NAMES[i]], EXAMPLES.shape[0])
    #
    # # Example tree node for textbook example
    # root = TreeNode("Patrons")
    # hungry = TreeNode("Hungry")
    # root.children = {"None": False, "Some": True, "Full": hungry}
    # typ = TreeNode("Type")
    # hungry.children = {1: typ, 0: False}
    # fri_sat = TreeNode("Fri/Sat")
    # typ.children = {"French": True, "Italian": False, "Thai": fri_sat, "Burger": True}
    # fri_sat.children = {0: False, 1: True}
    #
    # print(f'--------------Example Tree from Textbook--------------')
    # root.print_tree()
    #
    # print(f"Attribute count: {EXAMPLES.shape[1]}")
    # print(f"Example count: {EXAMPLES.shape[0]}")

    # subset = get_examples_by_type(EXAMPLES, "$$$", attributes["Price"].col_idx)
    # print(subset)
    #
    # best = choose_best_attribute(attributes, EXAMPLES)
    # print(best.name)

    # Build tree through "learning"
    print(EXAMPLES[1, -1])
    print(bool(int(EXAMPLES[1, -1])))

    root = DTL(EXAMPLES, attributes, None)
    root.print_tree()