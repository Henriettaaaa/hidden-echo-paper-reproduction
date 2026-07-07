import unittest

import torch
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForSequenceClassification

from modeling.my.configuration import MyQwen2Config
from modeling.my.split_echoslim import (
    CompactSideTransformerStack,
    SplittedQwen2ForSequenceClassification,
)


def make_tiny_config(**overrides):
    defaults = dict(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
        num_labels=3,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        use_cache=False,
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
    return MyQwen2Config(**defaults)


def make_split_model(**config_overrides):
    torch.manual_seed(1234)
    config = make_tiny_config(**config_overrides)
    base_model = Qwen2ForSequenceClassification(config)
    model = SplittedQwen2ForSequenceClassification.from_qwen2(base_model)
    model.eval()
    return model


class EchoSlimStructureTest(unittest.TestCase):
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
        self.assertIsInstance(trans, CompactSideTransformerStack)
        self.assertEqual(trans.selected_layer_indices, [1, 2, 3])
        self.assertEqual(len(trans.dec_layers), 3)
        self.assertEqual(len(trans.gate_vectors.gate_vectors), 3)
        self.assertEqual(len(model.mi_estimators), 3)

    def test_forward_after_layer_selection_returns_finite_classification_output(self):
        model = make_split_model(auto_skip=True, lst_skip=[-1])
        model.set_layer_skip([2, 1, 0, 3], num_reserved_layers=2, keep_last_layer=True)

        input_ids = torch.tensor(
            [
                [5, 6, 7, 0, 0],
                [8, 9, 10, 11, 0],
            ],
            dtype=torch.long,
        )
        attention_mask = (input_ids != 0).long()
        inputs_embeds = torch.randn(input_ids.shape[0], input_ids.shape[1], 16)
        labels = torch.tensor([0, 2], dtype=torch.long)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                return_dict=True,
            )

        self.assertEqual(tuple(outputs.logits.shape), (2, 3))
        self.assertTrue(torch.isfinite(outputs.logits).all().item())
        self.assertIsNotNone(outputs.loss)
        self.assertTrue(torch.isfinite(outputs.loss).item())
        self.assertGreater(model.total_embedding_data_transferred, 0)
        self.assertGreater(model.total_hidden_states_data_transferred, 0)

    def test_selected_backbone_layer_initializes_matching_compact_block(self):
        model = make_split_model(auto_skip=True, lst_skip=[-1])
        model.set_layer_skip([1, 3, 0, 2], num_reserved_layers=1, keep_last_layer=True)

        self.assertEqual(model.selected_layer_indices, [1, 3])
        trans = model.client_denoise.ladder_side.trans

        compact_q_proj = trans.dec_layers[0].self_attn.q_proj.weight
        source_q_proj = model.server_backbone.layers[1].self_attn.q_proj.weight
        self.assertTrue(
            torch.allclose(
                compact_q_proj,
                source_q_proj[: compact_q_proj.shape[0], : compact_q_proj.shape[1]],
            )
        )

        for gate in trans.gate_vectors.gate_vectors:
            self.assertTrue(torch.count_nonzero(gate).item() == 0)

    def test_edge_layer_selection_modes(self):
        no_last_model = make_split_model(auto_skip=True, lst_skip=[-1])
        no_last_model.set_layer_skip(
            [2, 1, 0, 3], num_reserved_layers=1, keep_last_layer=False
        )
        self.assertEqual(no_last_model.selected_layer_indices, [2])
        self.assertEqual(len(no_last_model.client_denoise.ladder_side.trans.dec_layers), 1)
        self.assertEqual(len(no_last_model.mi_estimators), 1)

        keep_last_model = make_split_model(auto_skip=True, lst_skip=[-1])
        keep_last_model.set_layer_skip(
            [1, 0, 2, 3], num_reserved_layers=1, keep_last_layer=True
        )
        self.assertEqual(keep_last_model.selected_layer_indices, [1, 3])
        self.assertEqual(len(keep_last_model.client_denoise.ladder_side.trans.dec_layers), 2)
        self.assertEqual(len(keep_last_model.mi_estimators), 2)

        manual_model = make_split_model(
            auto_skip=False,
            lst_skip=[1],
            mi_downsample_enable=False,
        )
        self.assertEqual(manual_model.selected_layer_indices, [0, 2, 3])
        self.assertEqual(manual_model.server_layer_select.lst_skip, [1])
        self.assertEqual(len(manual_model.client_denoise.ladder_side.trans.dec_layers), 3)

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

        self.assertEqual(len(full_model.client_denoise.ladder_side.trans.dec_layers), 4)
        self.assertEqual(len(slim_model.client_denoise.ladder_side.trans.dec_layers), 1)
        self.assertLess(slim_model._calc_client_params(), full_model._calc_client_params())

    def test_selection_mismatch_fails_fast(self):
        model = make_split_model(auto_skip=True, lst_skip=[-1], mi_downsample_enable=False)
        model.set_layer_skip([2, 1, 0, 3], num_reserved_layers=1, keep_last_layer=False)

        model.server_layer_select.lst_skip = [0, 1, 2, 3]
        input_ids = torch.tensor([[5, 6, 7, 0]], dtype=torch.long)
        attention_mask = (input_ids != 0).long()
        inputs_embeds = torch.randn(input_ids.shape[0], input_ids.shape[1], 16)

        with self.assertRaisesRegex(RuntimeError, "expected selected layer 2"):
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                labels=torch.tensor([1], dtype=torch.long),
                return_dict=True,
            )


if __name__ == "__main__":
    unittest.main()
