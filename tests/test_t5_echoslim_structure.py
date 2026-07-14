import unittest

import torch
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration

from modeling.my_t5.configuration import MyT5Config
from modeling.my_t5.split_echoslim import (
    CompactSideT5Stack,
    SplittedT5ForConditionalGeneration,
)


def make_tiny_config(**overrides):
    defaults = dict(
        vocab_size=64,
        d_model=16,
        d_kv=4,
        d_ff=32,
        num_layers=4,
        num_decoder_layers=4,
        num_heads=2,
        dropout_rate=0.0,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
        lst_enable=True,
        lst_reduce_factor=2,
        lst_skip=[-1],
        lst_input_type="clean",
        lst_random_init=False,
        auto_skip=True,
        num_reserved_layers=2,
        keep_last_layer=True,
        privacy_budget=5000.0,
        clip_embedding_l2=False,
        mi_downsample_enable=True,
        mi_estimator_hidden_dim=8,
        mi_estimator_iter_num=1,
    )
    defaults.update(overrides)
    return MyT5Config(**defaults)


def make_split_model(**config_overrides):
    torch.manual_seed(1234)
    config = make_tiny_config(**config_overrides)
    base_model = T5ForConditionalGeneration(config)
    model = SplittedT5ForConditionalGeneration.from_t5(base_model)
    model.eval()
    return model


class T5EchoSlimStructureTest(unittest.TestCase):
    def test_auto_skip_defers_client_denoiser_until_layer_selection(self):
        model = make_split_model(auto_skip=True, lst_skip=[-1])

        self.assertIsNone(model.client_denoise)
        self.assertIsNone(model.selected_layer_indices)
        self.assertEqual(model.mi_estimators, [])

        model.set_layer_skip([2, 1, 0, 3], num_reserved_layers=2, keep_last_layer=True)

        self.assertEqual(model.selected_layer_indices, [1, 2, 3])
        self.assertEqual(model.config.lst_skip, [0])
        self.assertEqual(model.server_layer_select.lst_skip, [0])

        trans = model.client_denoise.ladder_side.trans
        self.assertIsInstance(trans, CompactSideT5Stack)
        self.assertEqual(trans.selected_layer_indices, [1, 2, 3])
        self.assertEqual(len(trans.block), 3)
        self.assertEqual(len(trans.gate_vectors.gate_vectors), 3)
        self.assertEqual(len(model.mi_estimators), 3)

    def test_forward_after_layer_selection_returns_finite_generation_output(self):
        model = make_split_model(auto_skip=True, lst_skip=[-1])
        model.set_layer_skip([2, 1, 0, 3], num_reserved_layers=2, keep_last_layer=True)

        batch_size = 2
        seq_len = 5
        input_ids = torch.tensor(
            [
                [5, 6, 7, 0, 0],
                [8, 9, 10, 11, 0],
            ],
            dtype=torch.long,
        )
        attention_mask = (input_ids != 0).long()
        inputs_embeds = torch.randn(batch_size, seq_len, 16)
        labels = torch.tensor(
            [
                [12, 13, 1, -100],
                [14, 15, 16, 1],
            ],
            dtype=torch.long,
        )

        with torch.no_grad():
            outputs = model(
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                return_dict=True,
            )

        self.assertEqual(tuple(outputs.logits.shape[:2]), (batch_size, labels.shape[1]))
        self.assertTrue(torch.isfinite(outputs.logits).all().item())
        self.assertIsNotNone(outputs.loss)
        self.assertTrue(torch.isfinite(outputs.loss).item())
        self.assertGreater(model.total_embedding_data_transferred, 0)
        self.assertGreater(model.total_hidden_states_data_transferred, 0)

    def test_selected_encoder_layer_initializes_matching_compact_block(self):
        model = make_split_model(auto_skip=True, lst_skip=[-1])
        model.set_layer_skip([1, 3, 0, 2], num_reserved_layers=1, keep_last_layer=True)

        self.assertEqual(model.selected_layer_indices, [1, 3])
        trans = model.client_denoise.ladder_side.trans

        compact_q = trans.block[0].layer[0].SelfAttention.q.weight
        source_q = model.server_encoder.block[1].layer[0].SelfAttention.q.weight
        self.assertTrue(
            torch.allclose(
                compact_q,
                source_q[: compact_q.shape[0], : compact_q.shape[1]],
            )
        )

        for gate in trans.gate_vectors.gate_vectors:
            self.assertTrue(torch.count_nonzero(gate).item() == 0)

    def test_structural_sparsity_reduces_client_parameters(self):
        full_model = make_split_model(
            auto_skip=False,
            lst_skip=[],
            mi_downsample_enable=False,
        )
        slim_model = make_split_model(
            auto_skip=True,
            lst_skip=[-1],
            mi_downsample_enable=False,
        )
        slim_model.set_layer_skip(
            [2, 1, 0, 3], num_reserved_layers=1, keep_last_layer=False
        )

        self.assertEqual(len(full_model.client_denoise.ladder_side.trans.block), 4)
        self.assertEqual(len(slim_model.client_denoise.ladder_side.trans.block), 1)
        self.assertLess(slim_model._calc_client_params(), full_model._calc_client_params())

    def test_selection_mismatch_fails_fast(self):
        model = make_split_model(auto_skip=True, lst_skip=[-1], mi_downsample_enable=False)
        model.set_layer_skip([2, 1, 0, 3], num_reserved_layers=1, keep_last_layer=False)

        model.server_layer_select.lst_skip = [0, 1, 2, 3]
        input_ids = torch.tensor([[5, 6, 7, 0]], dtype=torch.long)
        attention_mask = (input_ids != 0).long()
        inputs_embeds = torch.randn(input_ids.shape[0], input_ids.shape[1], 16)
        labels = torch.tensor([[12, 13, 1]], dtype=torch.long)

        with self.assertRaisesRegex(RuntimeError, "expected selected layer 2"):
            model(
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                return_dict=True,
            )


if __name__ == "__main__":
    unittest.main()
