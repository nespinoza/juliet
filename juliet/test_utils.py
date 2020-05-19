import unittest
from .utils import *


class TestReverseLdCoeffs(unittest.TestCase):

    def test_reverse_ld_coeffs(self):
        pass


class TestReverseQCoeffs(unittest.TestCase):

    def test_quadratic(self):
        expected_q1 = 36.
        expected_q2 = 0.16666666666666666

        q1, q2 = reverse_q_coeffs("quadratic", 2., 4.)

        self.assertEqual(q1, expected_q1)
        self.assertEqual(q2, expected_q2)
