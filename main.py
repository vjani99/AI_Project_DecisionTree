from collections import namedtuple
import pandas as pd
import numpy as np

Child = namedtuple('Child', 'value entropy')

class TreeNode:
    children = {}


if __name__ == '__main__':
    data = pd.read_csv('data/figure_18-3.csv', sep=',').to_numpy()
    ATTRIBUTE_COUNT = data.shape[1] - 1

    examples = np.array(data[:, 1:ATTRIBUTE_COUNT], dtype=str)
    classifications = np.array(data[:, ATTRIBUTE_COUNT], dtype=int)

    print(examples)
    print(classifications)