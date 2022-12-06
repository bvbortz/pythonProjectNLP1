# -*- coding: utf-8 -*-
import numpy as np

def main():
    """
    Add Documentation here
    """
    hot = get_one_hot(5, 1)
    print(hot)
    pass  # Replace Pass with Your Code

def get_one_hot(size, ind):
    """
    this method returns a one-hot vector of the given size, where the 1 is placed in the ind entry.
    :param size: the size of the vector
    :param ind: the entry index to turn to 1
    :return: numpy ndarray which represents the one-hot vector
    """
    one_hot = np.zeros(size)
    one_hot[ind] = 1
    return one_hot

if __name__ == '__main__':
    main()
