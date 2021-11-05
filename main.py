import pandas as pd
import numpy as np
from sys import maxsize
from random import randint
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


def DTL(examples, attributes, default):
    """Implement a learning mode for decision tree via the DTL algorithm.
        :param examples - 2D Numpy array of training examples, with outcome in last column
        :param attributes - Dictionary of name:Attribute() pairs
        :param default - default value to return if no examples left to consider
    """
    if examples.size == 0:
        return default
    elif all_classes_same(examples):
        # Since we know all classes are same, just return classification of first example
        return bool(int(examples[0, -1]))
    elif not attributes:
        return mode(examples)
    else:
        # Update the frequencies of attribute types with new set of examples
        for a in attributes.values():
            a.update_counts(examples)

        best = choose_best_attribute(attributes, examples)
        tree = TreeNode(best.name)

        # Remove best attribute from dictionary (copy) for all subtrees.
        subset_attributes = attributes.copy()
        del subset_attributes[best.name]

        # Divide examples by the types of the best attribute.
        for val in best.type_names:
            ex_val_subset = get_examples_by_type(examples, val, best.col_idx)
            sub_tree = DTL(ex_val_subset, subset_attributes, mode(examples))
            tree.add_branch(val, sub_tree)
        return tree


def get_examples_by_type(examples, a_type, a_col_idx):
    """Get subset of examples that match this attribute type."""
    return examples[np.where(examples[:, a_col_idx] == a_type)]


def mode(examples):
    """Get most common outcome in the examples, either true or false."""
    outcomes = get_outcomes(examples)  # Extract column of all outcomes into a 1D array.
    counts = np.bincount(outcomes)  # Count occurrences of 0/1 in 1D array.

    # If tie in mode, randomly choose 1 or 0.
    if counts.size != 1 and counts[0] == counts[1]:
        mode = randint(0, 1)
    else:
        mode = counts.argmax()

    return bool(mode)  # Calculate mode and return as a boolean


def get_outcomes(examples):
    """Return outcomes of examples as a 1D array."""
    return np.array(examples[:, -1].T, np.int64)


def all_classes_same(examples):
    """Return whether the outcomes of each example are all the same"""
    outcomes = get_outcomes(examples)  # Extract row of all outcomes
    return np.all(outcomes == outcomes[0])  # Return whether all occurrences are the same


def choose_best_attribute(att_dict, examples):
    # Calculate info gain for each attribute and return the best.
    max_info_gain = -maxsize
    best_attribute = None

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
    """A data structure for a node in a directed decision tree
        :param name - String name to label current node with.
    """

    def __init__(self, name):
        """Initialize a node with a dictionary of children and it's name"""
        self.children = {}
        self.name = name

    def add_branch(self, value, node):
        """Map a node to a child of boolean value or a subtree root node"""
        self.children[value] = node

    def print_tree(self):
        """Print all nodes of tree (parent with children below)."""
        print("\n" + self.name + "?")
        print([f"{key}: Tree({c.name})" if isinstance(c, TreeNode)
               else f"{key}: {c}" for key, c in self.children.items()])

        # Traverse subtrees if any exist.
        for child in self.children.values():
            if isinstance(child, TreeNode):
                child.print_tree()


class Attribute:
    """Class to house details about a particular attribute and distribution of types.
        :param name - String name of attribute used for tree labeling.
        :param examples - 2D array of training examples.
        :param col_idx - Column in examples, corresponding to attribute values.
    """

    def __init__(self, name, examples, col_idx):
        self.name = name
        self.col_idx = col_idx

        # Find all unique types of attribute and initialize frequencies to 0.
        outputs = self.get_column(examples)
        self.type_names = np.unique(outputs)
        self.types = {a_type: Type() for a_type in self.type_names}

    def update_counts(self, examples):
        """Update frequencies of each output value corresponding to this set of examples."""
        outputs = self.get_column(examples)
        outcomes = get_outcomes(examples)

        # Reset type frequencies and recount with new set of examples
        self.reset_type_frequencies()
        # print(f'--------------{self.name} types: {self.type_names}--------------\n')
        for k, a_type in enumerate(outputs):
            if outcomes[k]:
                self.types[a_type].p += 1  # Positive
            else:
                self.types[a_type].n += 1  # Negative
            self.types[a_type].v += 1  # Total
        # print(f'{self.name} frequencies [p, k, v]: {self.type_count}\n')

    def reset_type_frequencies(self):
        """Clear the frequencies of each type corresponding to this attribute."""
        for t in self.types.values():
            t.reset()

    def get_column(self, examples):
        """Extract column of outputs related to this this attribute as a 1D array."""
        return examples[:, self.col_idx].T


class Type:
    """Data structure to hold positive, negative, and total counts of an attribute type."""

    def __init__(self):
        self.p = 0
        self.n = 0
        self.v = 0

    def reset(self):
        """Re-initialize all frequencies to 0."""
        self.__init__()


def goal_entropy(examples):
    """Calculate the entropy relative to the subset of examples/outcomes."""
    outcomes = get_outcomes(examples)  # Extract row of all outcomes
    counts = np.bincount(outcomes)

    # Calculate probability of true/false
    prob_p = counts[1] / examples.shape[0]
    prob_n = 1 - prob_p

    # Calculate and return goal entropy of this example subset.
    # Note: A H(goal) of 1 means that there are equal # of positive and negative examples
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.nan_to_num(-prob_p * np.log2(prob_p) - prob_n * np.log2(prob_n))


def aggregate_entropy(attribute, example_count):
    """Calculate the entropy values of each type of an attribute and return aggregate entropy."""
    weighted_sum = 0
    for k in attribute.type_names:
        # print(f'({attribute.name}, {k}):')
        vk = attribute.types[k].v
        if vk != 0:
            prob_p = attribute.types[k].p / vk
            prob_n = attribute.types[k].n / vk
            with np.errstate(divide='ignore', invalid='ignore'):  # TODO: Can we safely ignore these warnings?
                entropy = np.nan_to_num(-prob_p * np.log2(prob_p) - prob_n * np.log2(prob_n))
            # print(f'Probability (+, -): ({prob_p:.2f}, {prob_n: .2f})\nEntropy: {entropy:.2f}\n')

            # 'weight' the sum of the type entropies.
            weighted_sum += (vk / example_count) * entropy
    # print(f'* ({attribute.name}) Aggregate entropy: {weighted_sum}\n')
    return weighted_sum


def leave_one_out_validate(true_tree, examples):
    validation_runs = examples.shape[0]
    predictions = []
    test_labels = []
    for i in range(validation_runs):
        # Shuffle examples and exclude last element for validation set.
        np.random.shuffle(examples)
        validation_set = examples[:-1, :]
        test_example = examples[-1]
        print(f'Run {i + 1}, leaving out: {test_example}')

        # Learn on reduced set of examples.
        tree = DTL(validation_set, attributes_dict, None)

        # TODO: Predict unknown label using validation DT and store it
        predictions.append(bool(randint(0,1)))
        # predictions.append(predict_outcome(tree, test_example))

        # TODO: Get known label from original learned DT and store it
        test_labels.append(bool(randint(0, 1)))
        # test_labels.append(predict_outcome(true_tree, test_example))
    return np.array(predictions), np.array(test_labels)


if __name__ == '__main__':
    # Read in examples from formatted .csv file.
    data = pd.read_csv('data/figure_18-3.csv', sep=',', header=None).to_numpy()

    # Constants related to data read in.
    ATTRIBUTE_COUNT = data.shape[1] - 1
    EXAMPLE_COUNT = data.shape[0] - 1
    ATTRIBUTE_NAMES = np.array(data[0, 1:], dtype=str)

    # Read in all attributes, AND the outcome in the last index. Skips first row of table headers.
    EXAMPLES = np.array(data[1:, 1:ATTRIBUTE_COUNT + 1], dtype=str)
    ###### Special cases for testing ######
    # EXAMPLES = np.append(EXAMPLES, [EXAMPLES[0]], axis=0) # Unbalanced data set
    # print(EXAMPLES)
    # EXAMPLES = EXAMPLES[0]    # A single example data set
    ######################################
    EXAMPLES = np.atleast_2d(EXAMPLES)  # Ensure that a 1D array is still treated as 2D in all calculations.

    attributes_dict = {}
    for i in range(0, EXAMPLES.shape[1] - 1):
        attributes_dict[ATTRIBUTE_NAMES[i]] = Attribute(ATTRIBUTE_NAMES[i], EXAMPLES, i)

    # print(f'--------------Example Tree from Textbook--------------')
    # root = TreeNode("Patrons")
    # hungry = TreeNode("Hungry")
    # root.children = {"None": False, "Some": True, "Full": hungry}
    # typ = TreeNode("Type")
    # hungry.children = {1: typ, 0: False}
    # fri_sat = TreeNode("Fri/Sat")
    # typ.children = {"French": True, "Italian": False, "Thai": fri_sat, "Burger": True}
    # fri_sat.children = {0: False, 1: True}
    # root.print_tree()

    """---------------------- Learning mode: Construct tree using the DTL algorithm ----------------------"""
    root = DTL(EXAMPLES, attributes_dict, None)

    try:
        root.print_tree()
    except AttributeError:
        print(f"Single example encountered, defaulting to single outcome: {root}")

    true_outcomes, pred_outcomes = leave_one_out_validate(root, EXAMPLES)
    print(classification_report(true_outcomes, pred_outcomes))
    plt.figure()
