"""Per-case randomness and budget-prefix contracts without GPU/model downloads."""
import random
import unittest

import torch

from .case_rng import case_sampling, sampling_seed
from .length_probe import prefix_gate


class PerCaseRNG(unittest.TestCase):
    def test_stable_seed_has_explicit_axes_and_rejects_bad_ids(self):
        seed = sampling_seed(7, "题目/1", 2, 3)
        self.assertEqual(seed, sampling_seed(7, "题目/1", 2, 3))
        self.assertTrue(0 <= seed < 2**63)
        self.assertEqual(len({seed, sampling_seed(8, "题目/1", 2, 3),
                              sampling_seed(7, "题目/2", 2, 3),
                              sampling_seed(7, "题目/1", 3, 3),
                              sampling_seed(7, "题目/1", 2, 4)}), 5)
        with self.assertRaises(ValueError):
            sampling_seed(7, "", 0)

    def test_order_and_prior_output_length_do_not_shift_next_case(self):
        def sample(identifier, count):
            with case_sampling(17, identifier):
                return torch.rand(count)
        short = sample("first", 3)
        next_short = sample("next", 10)
        long = sample("first", 97)
        next_long = sample("next", 10)
        torch.testing.assert_close(short, long[:3], atol=0, rtol=0)
        torch.testing.assert_close(next_short, next_long, atol=0, rtol=0)
        sample("unrelated", 1000)
        torch.testing.assert_close(next_short, sample("next", 10), atol=0, rtol=0)

    def test_outer_rng_is_restored_even_after_error(self):
        cpu, python = torch.get_rng_state().clone(), random.getstate()
        with self.assertRaises(RuntimeError):
            with case_sampling(7, "failure"):
                torch.rand(50)
                random.random()
                raise RuntimeError("injected")
        self.assertTrue(torch.equal(cpu, torch.get_rng_state()))
        self.assertEqual(python, random.getstate())

    def test_nested_cases_do_not_change_the_parent_stream(self):
        with case_sampling(9, "outer"):
            first = torch.rand(2)
            with case_sampling(8, "inner"):
                torch.rand(83)
            second = torch.rand(2)
        with case_sampling(9, "outer"):
            expected = torch.rand(4)
        torch.testing.assert_close(torch.cat((first, second)), expected, atol=0, rtol=0)

    def test_budget_prefix_cannot_silently_change_the_answer(self):
        prefix_gate({"id": "x", "token_ids": [1, 2]}, {"id": "x", "token_ids": [1, 2, 3]})
        with self.assertRaises(ValueError):
            prefix_gate({"id": "x", "token_ids": [1, 2]}, {"id": "x", "token_ids": [1, 9, 3]})
        with self.assertRaises(ValueError):
            prefix_gate({"id": "x", "token_ids": [1]}, {"id": "y", "token_ids": [1]})


if __name__ == "__main__":
    unittest.main()
