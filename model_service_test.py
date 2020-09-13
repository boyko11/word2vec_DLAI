import unittest
from w2v_utils import read_glove_vecs
from model_service import ModelService
import numpy as np


class ModelServiceTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.words, cls.word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')
        cls.model_service = ModelService()

    def test_cosine_similarity(self):

        father = self.word_to_vec_map["father"]
        mother = self.word_to_vec_map["mother"]
        ball = self.word_to_vec_map["ball"]
        crocodile = self.word_to_vec_map["crocodile"]
        france = self.word_to_vec_map["france"]
        italy = self.word_to_vec_map["italy"]
        paris = self.word_to_vec_map["paris"]
        rome = self.word_to_vec_map["rome"]

        father_mother_sim = self.model_service.cosine_similarity(father, mother)
        ball_croc_sim = self.model_service.cosine_similarity(ball, crocodile)
        fr_to_paris_as_rome_to_it = self.model_service.cosine_similarity(france - paris, rome - italy)

        print("cosine_similarity(father, mother) = ", father_mother_sim)
        print("cosine_similarity(ball, crocodile) = ", ball_croc_sim)
        print("cosine_similarity(france - paris, rome - italy) = ", fr_to_paris_as_rome_to_it)

        self.assertEqual(0.891, np.round(father_mother_sim, 3))
        self.assertEqual(0.274, np.round(ball_croc_sim, 3))
        self.assertEqual(-0.675, np.round(fr_to_paris_as_rome_to_it, 3))

    def test_complete_analogy(self):

        triads_to_try = [('italy', 'italian', 'spain'), ('india', 'delhi', 'japan'), ('man', 'woman', 'boy'),
                         ('small', 'smaller', 'large')]
        for triad in triads_to_try:
            print('{} -> {} :: {} -> {}'.format(*triad,
                                                self.model_service.complete_analogy(*triad, self.word_to_vec_map)))


if __name__ == '__main__':
    unittest.main()
