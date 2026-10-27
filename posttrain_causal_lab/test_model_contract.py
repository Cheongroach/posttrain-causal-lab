"""A tiny native Qwen3.5 integration check; no published weights are loaded."""
import unittest

import torch
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
from transformers.models.qwen3_5 import modeling_qwen3_5

from .contracts import output_intervention, token_loss
from .pilot import adapter_state, attach, force_torch_delta, restore_adapter


class TinyModelContract(unittest.TestCase):
    def test_hybrid_lora_backward_intervention_and_adapter_removal(self):
        torch.manual_seed(20261027)
        modeling_qwen3_5.FusedRMSNormGated = None
        config = Qwen3_5TextConfig(
            vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
            num_attention_heads=2, num_key_value_heads=1, head_dim=16,
            linear_num_key_heads=2, linear_num_value_heads=2,
            linear_key_head_dim=8, linear_value_head_dim=8,
            layer_types=["linear_attention"] * 3 + ["full_attention"],
            rope_parameters={"rope_type": "default", "rope_theta": 10000.,
                             "partial_rotary_factor": 0.5, "mrope_section": [1, 1, 2],
                             "mrope_interleaved": True})
        base = Qwen3_5ForCausalLM(config)
        force_torch_delta(base)
        student = attach(base)
        state = adapter_state(student)
        ids = torch.tensor([[4, 5, 6, 7, 8, 9]])

        def forward():
            return student(input_ids=ids, attention_mask=torch.ones_like(ids),
                           logits_to_keep=torch.tensor([2, 3, 4]), use_cache=False).logits

        before = forward().detach()
        gdn = base.model.layers[0].linear_attn
        with output_intervention(gdn, "identity"):
            torch.testing.assert_close(forward(), before, rtol=0, atol=0)
        with output_intervention(gdn, "zero"):
            self.assertGreater(float((forward().detach() - before).abs().max()), 1e-7)
        optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=.01)
        loss = token_loss(forward(), ids[:, 3:], torch.ones(1, 3, dtype=torch.bool))
        loss.backward()
        gradients = [p.grad for p in student.parameters() if p.requires_grad and p.grad is not None]
        self.assertTrue(all(bool(torch.isfinite(g).all()) for g in gradients))
        self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0.)
        optimizer.step()
        self.assertGreater(float((forward().detach() - before).abs().max()), 1e-7)
        with student.disable_adapter():
            torch.testing.assert_close(forward(), before, atol=0, rtol=0)
        restore_adapter(student, state)
        torch.testing.assert_close(forward(), before, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
