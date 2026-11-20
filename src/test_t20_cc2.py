"""Synthetic numerical tests for T20 conditional LEACE, float64 only."""
import unittest
import numpy as np

from leace_cc2 import fit_cc2, transform_cc2, leace_covariance


def conditional_covariance(x, z, w, group):
    mask = w == group
    xx, zz = x[mask], z[mask]
    return (xx - xx.mean(0)).T @ (zz - zz.mean()) / len(zz)


class TestCC2(unittest.TestCase):
    def data(self, collinear=False, n=4000):
        rng = np.random.default_rng(231)
        w = rng.integers(0, 2, n)
        z = (rng.random(n) < np.where(w == 0, .3, .7)).astype(float)
        x = rng.normal(0, .2, (n, 4))
        # W has an independent encoding direction. Z occupies one or two
        # different conditional directions; group means alone are insufficient.
        x[:, 2] += 4 * w
        x[:, 0] += z * (w == 0)
        x[:, 0 if collinear else 1] += z * (w == 1)
        return x, z, w

    def test_rank_two_covariance_idempotence_and_global_covariance(self):
        x, z, w = self.data()
        fit = fit_cc2(x, z, w)
        xp = transform_cc2(x, fit)
        self.assertEqual(fit["rank"], 2)
        for group in (0, 1):
            self.assertLess(np.linalg.norm(conditional_covariance(xp, z, w, group)), 1e-8)
        self.assertGreater(np.linalg.norm((xp - xp.mean(0)).T @ (z - z.mean()) / len(z)), 1e-3)
        self.assertLess(np.max(np.abs(transform_cc2(xp, fit) - xp)), 1e-8)
        before = x[w == 1].mean(0) - x[w == 0].mean(0)
        after = xp[w == 1].mean(0) - xp[w == 0].mean(0)
        cosine = before @ after / (np.linalg.norm(before) * np.linalg.norm(after))
        self.assertGreater(cosine, .9)

    def test_collinear_rank_one_equivalence(self):
        x, z, w = self.data(collinear=True)
        # Remove finite-sample correlations of nuisance columns with Z so
        # both empirical conditional covariance vectors are exactly parallel.
        for group in (0, 1):
            mask = w == group
            zz = z[mask] - z[mask].mean()
            for col in (1, 2, 3):
                x[mask, col] -= zz * np.dot(x[mask, col], zz) / np.dot(zz, zz)
        fit = fit_cc2(x, z, w)
        self.assertEqual(fit["rank"], 1)
        x1 = transform_cc2(x, fit)
        c0 = conditional_covariance(x, z, w, 0)
        v = np.linalg.solve(leace_covariance(x), c0)
        rank_one = np.outer(c0, v) / (c0 @ v)
        self.assertLess(np.max(np.abs(fit["removal"] - rank_one)), 1e-8)
        for group in (0, 1):
            self.assertLess(np.linalg.norm(conditional_covariance(x1, z, w, group)), 1e-8)
        self.assertTrue(fit["near_collinear"])

    def test_zero_direction_and_missing_group(self):
        x, z, w = self.data()
        # The target still varies, but one group's empirical covariance is 0.
        mask = w == 0
        zz = z[mask] - z[mask].mean()
        for col in range(x.shape[1]):
            x[mask, col] -= zz * np.dot(x[mask, col], zz) / np.dot(zz, zz)
        fit = fit_cc2(x, z, w)
        self.assertEqual(fit["rank"], 1)
        self.assertLess(max(fit["residual_norms"]), 1e-8)
        for group in (0, 1):
            mask = w == group
            zz = z[mask] - z[mask].mean()
            for col in range(x.shape[1]):
                x[mask, col] -= zz * np.dot(x[mask, col], zz) / np.dot(zz, zz)
        none = fit_cc2(x, z, w)
        self.assertEqual(none["rank"], 0)
        self.assertLess(np.max(np.abs(transform_cc2(x, none) - x)), 1e-12)
        # A constant target in an observed stratum has a defined zero
        # empirical conditional covariance. If both strata are constant,
        # the estimator returns the identity and records the degeneracy.
        none = fit_cc2(x, w.astype(float), w)
        self.assertEqual(none["rank"], 0)
        self.assertEqual(none["degenerate_groups"], [0, 1])
        self.assertLess(np.max(np.abs(transform_cc2(x, none) - x)), 1e-12)
        with self.assertRaises(ValueError):
            fit_cc2(x, z, np.zeros_like(w))

    def test_constant_target_in_one_stratum_is_zero_covariance(self):
        x, z, w = self.data()
        z[w == 0] = 0
        fit = fit_cc2(x, z, w)
        self.assertEqual(fit["degenerate_groups"], [0])
        self.assertEqual(fit["rank"], 1)
        xp = transform_cc2(x, fit)
        self.assertLess(np.linalg.norm(conditional_covariance(xp, z, w, 0)), 1e-12)
        self.assertLess(np.linalg.norm(conditional_covariance(xp, z, w, 1)), 1e-8)

    def test_near_collinear_keeps_second_direction_when_residual_matters(self):
        x, z, w = self.data(collinear=True)
        for group in (0, 1):
            mask = w == group
            zz = z[mask] - z[mask].mean()
            for col in (1, 2, 3):
                x[mask, col] -= zz * np.dot(x[mask, col], zz) / np.dot(zz, zz)
        x[w == 1, 1] += 1e-7 * z[w == 1]
        fit = fit_cc2(x, z, w)
        self.assertTrue(fit["near_collinear"])
        self.assertEqual(fit["rank"], 2)
        self.assertLess(max(fit["residual_norms"]), 1e-8)

    def test_small_nonparallel_directions_still_have_rank_two(self):
        x, z, w = self.data()
        for group in (0, 1):
            mask = w == group
            zz = z[mask] - z[mask].mean()
            for col in range(x.shape[1]):
                x[mask, col] -= zz * np.dot(x[mask, col], zz) / np.dot(zz, zz)
            x[mask, group] += 1e-8 * z[mask]
        fit = fit_cc2(x, z, w)
        self.assertFalse(fit["near_collinear"])
        self.assertEqual(fit["rank"], 2)


if __name__ == "__main__":
    unittest.main()
