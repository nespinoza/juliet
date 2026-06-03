import unittest
from .utils import *


class TestDepthConversions(unittest.TestCase):

    def test_p_to_depth(self):
        self.assertAlmostEqual(p_to_depth(0.2), 0.04)
        self.assertEqual(p_to_depth(0.), 0.)
        self.assertAlmostEqual(p_to_depth(-0.2), -0.04)

    def test_depth_to_p(self):
        self.assertAlmostEqual(depth_to_p(0.04), 0.2)
        self.assertEqual(depth_to_p(0.), 0.)
        self.assertAlmostEqual(depth_to_p(-0.04), -0.2)

    def test_add_depth_derived_parameters(self):
        posterior_samples = {
            'depth_p1': np.array([0.01, 0.04, -0.09])
        }

        add_depth_derived_parameters(posterior_samples)

        np.testing.assert_allclose(
            posterior_samples['p_p1'],
            np.array([0.1, 0.2, -0.3])
        )

    def test_depth_prior_counts_transiting_planet_once(self):
        priors = {
            'b_p1': {
                'distribution': 'uniform',
                'hyperparameters': [0., 1.]
            },
            'depth_p1': {
                'distribution': 'uniform',
                'hyperparameters': [0., 0.1]
            }
        }

        n_transit, _, numbering_transit, _, _ = readpriors(priors)

        self.assertEqual(n_transit, 1)
        np.testing.assert_array_equal(numbering_transit, np.array([1]))


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
