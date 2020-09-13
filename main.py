import numpy as np
from w2v_utils import *


if __name__ == '__main__':

    words, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')
    print('Done.')

