"""Contracts specific to the paired scale experiment."""
import unittest

import torch

from .scale_contracts import cell_layout, choose_sites, grade, matched_random
from .scale24 import capture_sites, patch_site


class ScaleContracts(unittest.TestCase):
    def test_layout_owns_all_24_gpus_without_overlap(self):
        cells = [cell for worker in range(3) for cell in cell_layout(worker)]
        self.assertEqual(len(cells), 12)
        self.assertEqual(len({(cell["arm"], cell["seed"]) for cell in cells}), 12)
        for worker in range(3):
            self.assertEqual(sorted(gpu for cell in cell_layout(worker) for gpu in cell["gpus"]),
                             list(range(8)))
        with self.assertRaises(ValueError):
            cell_layout(3)

    def test_format_and_numeric_content_are_separate(self):
        marked = grade("The answer is #### 1,250", "1250", False)
        plain = grade("The answer is 1250.", "1250", False)
        self.assertTrue(marked["strict_correct"])
        self.assertTrue(plain["numeric_correct"])
        self.assertFalse(plain["format_valid"])
        self.assertFalse(plain["strict_correct"])
        self.assertTrue(grade("#### -3.50", "-3.5", False)["strict_correct"])
        self.assertFalse(grade("no number", "3", False)["numeric_correct"])

    def test_truncation_is_never_counted_as_complete_accuracy(self):
        row = grade("#### 42", "42", True)
        self.assertTrue(row["numeric_correct"])
        self.assertFalse(row["complete_numeric_correct"])
        self.assertFalse(row["strict_correct"])

    def test_random_intervention_matches_norm_reproducibly(self):
        source = torch.randn(1, 6, 16)
        donor = source + torch.randn_like(source)
        first = matched_random(source, donor, 17)
        torch.testing.assert_close(first, matched_random(source, donor, 17), atol=0, rtol=0)
        torch.testing.assert_close((first - source).norm(), (donor - source).norm(),
                                   atol=1e-5, rtol=1e-6)
        torch.testing.assert_close(matched_random(source, source, 17), source, atol=0, rtol=0)
        with self.assertRaises(ValueError):
            matched_random(source, donor[:, :-1], 17)

    def test_capture_and_patch_preserve_attention_tuple_and_cleanup(self):
        class Attention(torch.nn.Module):
            def forward(self, x):
                return x * 2, None
        module = Attention()
        inputs = torch.randn(1, 3, 5)
        captured = {}
        with capture_sites({"attention": module}, captured):
            original, weights = module(inputs)
        self.assertIsNone(weights)
        with patch_site(module, captured["attention"]):
            patched, weights = module(inputs * 4)
        self.assertIsNone(weights)
        torch.testing.assert_close(patched, original, atol=0, rtol=0)
        with self.assertRaises(RuntimeError):
            with patch_site(module, captured["attention"]):
                raise RuntimeError("injected failure")
        self.assertEqual(len(module._forward_hooks), 0)

    def test_selection_uses_only_mean_gold_drop_with_fixed_tie_break(self):
        records = [
            {"site": "b", "mode": "base", "gold_logp_drop": 3.},
            {"site": "a", "mode": "base", "gold_logp_drop": 1.},
            {"site": "a", "mode": "base", "gold_logp_drop": 5.},
            {"site": "c", "mode": "base", "gold_logp_drop": -1.},
            {"site": "c", "mode": "random", "gold_logp_drop": 100.},
        ]
        self.assertEqual(choose_sites(records, 2), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
