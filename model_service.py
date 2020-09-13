import numpy as np

class ModelService:

    def __init__(self):
        pass

    # GRADED FUNCTION: cosine_similarity
    def cosine_similarity(self, u, v):
        """
        Cosine similarity reflects the degree of similarity between u and v

        Arguments:
            u -- a word vector of shape (n,)
            v -- a word vector of shape (n,)

        Returns:
            cosine_similarity -- the cosine similarity between u and v defined by the formula above.
        """

        distance = 0.0

        ### START CODE HERE ###
        # Compute the dot product between u and v (≈1 line)
        dot = np.dot(u, v)
        # Compute the L2 norm of u (≈1 line)
        norm_u = np.sqrt(np.sum(np.square(u)))

        # Compute the L2 norm of v (≈1 line)
        norm_v = np.sqrt(np.sum(np.square(v)))
        # Compute the cosine similarity defined by formula (1) (≈1 line)
        cosine_similarity = dot / (norm_u * norm_v)
        ### END CODE HERE ###

        return cosine_similarity

    # GRADED FUNCTION: complete_analogy
    def complete_analogy(self, word_a, word_b, word_c, word_to_vec_map):
        """
        Performs the word analogy task as explained above: a is to b as c is to ____.

        Arguments:
        word_a -- a word, string
        word_b -- a word, string
        word_c -- a word, string
        word_to_vec_map -- dictionary that maps words to their corresponding vectors.

        Returns:
        best_word --  the word such that v_b - v_a is close to v_best_word - v_c, as measured by cosine similarity
        """

        # convert words to lowercase
        word_a, word_b, word_c = word_a.lower(), word_b.lower(), word_c.lower()

        ### START CODE HERE ###
        # Get the word embeddings e_a, e_b and e_c (≈1-3 lines)
        e_a, e_b, e_c = word_to_vec_map[word_a], word_to_vec_map[word_b], word_to_vec_map[word_c]
        ### END CODE HERE ###

        words = word_to_vec_map.keys()
        max_cosine_sim = -100  # Initialize max_cosine_sim to a large negative number
        best_word = None  # Initialize best_word with None, it will help keep track of the word to output

        # to avoid best_word being one of the input words, skip the input words
        # place the input words in a set for faster searching than a list
        # We will re-use this set of input words inside the for-loop
        input_words_set = set([word_a, word_b, word_c])

        # loop over the whole word vector set
        for w in words:
            # to avoid best_word being one of the input words, skip the input words
            if w in input_words_set:
                continue

            ### START CODE HERE ###
            # Compute cosine similarity between the vector (e_b - e_a) and the vector ((w's vector representation) - e_c)  (≈1 line)
            cosine_sim = self.cosine_similarity(e_b - e_a, word_to_vec_map[w] - e_c)

            # If the cosine_sim is more than the max_cosine_sim seen so far,
            # then: set the new max_cosine_sim to the current cosine_sim and the best_word to the current word (≈3 lines)
            if cosine_sim > max_cosine_sim:
                max_cosine_sim = cosine_sim
                best_word = w
            ### END CODE HERE ###

        return best_word