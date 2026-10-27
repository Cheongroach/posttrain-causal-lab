"""Small mathematical contracts; these do not validate the full GPU runtime."""
import copy
import unittest

import torch

from .contracts import output_intervention, response_mask, token_loss


class TrainingContracts(unittest.TestCase):
    def test_first_eos_is_supervised_and_padding_is_excluded(self):
        tokens = torch.tensor([[4, 9, 9, 9], [4, 5, 6, 7]])
        mask = response_mask(tokens, [9])
        self.assertEqual(mask.tolist(), [[True, True, False, False], [True] * 4])
        logits = torch.randn(2, 4, 10, requires_grad=True)
        token_loss(logits, tokens, mask).backward()
        self.assertEqual(float(logits.grad[0, 2:].abs().sum()), 0.0)
        self.assertGreater(float(logits.grad[0, 1].abs().sum()), 0.0)

    def test_kl_direction_token_mean_and_teacher_stop_gradient(self):
        student = torch.tensor([[[0., 1.], [20., -20.]]], requires_grad=True)
        teacher = torch.tensor([[[2., 0.], [-20., 20.]]], requires_grad=True)
        mask = torch.tensor([[True, False]])
        loss = token_loss(student, torch.tensor([[0, 1]]), mask, teacher)
        p = teacher[0, 0].detach().softmax(-1)
        expected = (p * (p.log() - student[0, 0].log_softmax(-1))).sum()
        torch.testing.assert_close(loss, expected)
        loss.backward()
        torch.testing.assert_close(student.grad[0, 0], student[0, 0].detach().softmax(-1) - p)
        self.assertIsNone(teacher.grad)
        self.assertEqual(float(student.grad[0, 1].abs().sum()), 0.0)

    def test_equal_distributions_have_zero_kl(self):
        x = torch.randn(1, 3, 7, requires_grad=True)
        mask = torch.ones(1, 3, dtype=torch.bool)
        loss = token_loss(x, torch.zeros(1, 3, dtype=torch.long), mask, x.detach())
        self.assertLess(abs(float(loss.detach())), 1e-7)
        loss.backward()
        self.assertLess(float(x.grad.abs().max()), 1e-7)

    def test_empty_masks_and_wrong_event_space_are_rejected(self):
        x = torch.zeros(1, 2, 3)
        with self.assertRaises(ValueError):
            token_loss(x, torch.zeros(1, 2, dtype=torch.long), torch.zeros(1, 2, dtype=torch.bool))
        with self.assertRaises(ValueError):
            token_loss(x, torch.zeros(1, 2, dtype=torch.long),
                       torch.ones(1, 2, dtype=torch.bool), torch.zeros(1, 2, 4))

    def test_identity_zero_donor_and_hook_cleanup(self):
        layer = torch.nn.Linear(3, 2)
        x = torch.randn(1, 4, 3)
        original = layer(x)
        with output_intervention(layer, "identity"):
            torch.testing.assert_close(layer(x), original, rtol=0, atol=0)
        with output_intervention(layer, "zero"):
            self.assertEqual(float(layer(x).detach().abs().sum()), 0.0)
        with output_intervention(layer, "replace", original):
            torch.testing.assert_close(layer(x * 3), original, rtol=0, atol=0)
        with self.assertRaises(RuntimeError):
            with output_intervention(layer, "zero"):
                raise RuntimeError("injected failure")
        self.assertEqual(len(layer._forward_hooks), 0)
        torch.testing.assert_close(layer(x), original, rtol=0, atol=0)

    def test_optimizer_state_is_necessary_for_identical_replay(self):
        torch.manual_seed(20261027)
        layer = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(layer.parameters(), lr=0.01)
        x = torch.randn(5, 3)

        def step():
            optimizer.zero_grad()
            loss = layer(x).square().mean()
            loss.backward()
            optimizer.step()
            return loss.detach()

        step()
        saved_model = copy.deepcopy(layer.state_dict())
        saved_optimizer = copy.deepcopy(optimizer.state_dict())
        reference_loss = step()
        reference_model = copy.deepcopy(layer.state_dict())
        layer.load_state_dict(saved_model)
        optimizer.load_state_dict(saved_optimizer)
        torch.testing.assert_close(step(), reference_loss, rtol=0, atol=0)
        for key, value in layer.state_dict().items():
            torch.testing.assert_close(value, reference_model[key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
