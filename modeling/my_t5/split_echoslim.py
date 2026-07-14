from copy import deepcopy
import re
from typing import Iterable, List, Optional, Tuple, Union
import weakref

from torch import nn
import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from typing_extensions import override
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.models.t5.modeling_t5 import (
    T5Block,
    T5ForConditionalGeneration,
    T5LayerNorm,
    T5PreTrainedModel,
    T5Stack,
)

from modeling.my_t5.configuration import MyT5Config
from modeling.my.split import HiddenDowns, LayerSelect, MINE
from utils.noise import get_noisy_embedding


class GateWrapper(nn.Module):
    def __init__(self, config: MyT5Config):
        super().__init__()
        self.config = config
        self.gate_vectors = nn.ParameterList(
            [nn.Parameter(torch.zeros(config.d_model)) for _ in range(config.num_layers)]
        )

    def forward(self, layer_idx: int):
        return torch.sigmoid(
            self.gate_vectors[layer_idx] / self.config.lst_temperature
        )


class CompactGateWrapper(nn.Module):
    def __init__(self, config: MyT5Config, selected_layer_indices: List[int]):
        super().__init__()
        self.config = config
        self.selected_layer_indices = list(selected_layer_indices)
        self.layer_to_compact_idx = {
            layer_idx: compact_idx
            for compact_idx, layer_idx in enumerate(self.selected_layer_indices)
        }
        self.gate_vectors = nn.ParameterList(
            [nn.Parameter(torch.zeros(config.d_model)) for _ in self.selected_layer_indices]
        )

    def forward(self, original_layer_idx: int):
        compact_idx = self.layer_to_compact_idx[original_layer_idx]
        return torch.sigmoid(
            self.gate_vectors[compact_idx] / self.config.lst_temperature
        )


class T5EncoderWithStartLayer(T5PreTrainedModel):
    def __init__(self, config: MyT5Config, encoder: T5Stack):
        super().__init__(config)
        self.embed_tokens = encoder.embed_tokens
        self.is_decoder = False
        self.block = encoder.block
        self.final_layer_norm = encoder.final_layer_norm
        self.dropout = encoder.dropout
        self.gradient_checkpointing = False
        self.model_parallel = False

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        past_key_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        *,
        start_layer: int = 0,
    ):
        use_cache = False
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds")
        if input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size, seq_length, device=inputs_embeds.device
            )
        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_shape
        )
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)

        hidden_states = inputs_embeds
        if start_layer == 0:
            hidden_states = self.dropout(hidden_states)

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        position_bias = None

        for i, (layer_module, past_key_value) in enumerate(
            zip(self.block[start_layer:], past_key_values[start_layer:]),
            start=start_layer,
        ):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                layer_head_mask=head_mask[i],
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]
            hidden_states = layer_outputs[0]
            position_bias = layer_outputs[2]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class SideT5Stack(T5PreTrainedModel):
    def __init__(self, config: MyT5Config):
        super().__init__(config)
        self.is_decoder = False
        self.block = nn.ModuleList(
            [
                T5Block(config, has_relative_attention_bias=bool(i == 0))
                for i in range(config.num_layers)
            ]
        )
        self.final_layer_norm = T5LayerNorm(
            config.d_model, eps=config.layer_norm_epsilon
        )
        self.dropout = nn.Dropout(config.dropout_rate)
        self.gate_vectors = GateWrapper(config)
        self.gradient_checkpointing = False
        self.model_parallel = False

    def forward(
        self,
        inputs_embeds: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor],
        backbone_hidden_states: Tuple[torch.FloatTensor | None],
        output_hidden_states: Optional[bool] = None,
    ) -> BaseModelOutput:
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        input_shape = inputs_embeds.size()[:-1]
        batch_size, seq_length = input_shape
        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size, seq_length, device=inputs_embeds.device
            )
        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_shape
        )

        hidden_states = self.dropout(inputs_embeds)
        residual_hidden_states = hidden_states
        position_bias = None
        all_hidden_states = () if output_hidden_states else None

        assert len(backbone_hidden_states) == self.config.num_layers
        for layer_idx, layer_module in enumerate(self.block):
            if backbone_hidden_states[layer_idx] is None:
                continue

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            gate = self.gate_vectors(layer_idx)
            hidden_states = (
                backbone_hidden_states[layer_idx] * gate
                + hidden_states * (1 - gate)
            )

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                use_cache=False,
                output_attentions=False,
            )
            layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]
            hidden_states = layer_outputs[0]
            position_bias = layer_outputs[2]

            if self.config.use_residual:
                hidden_states = hidden_states + residual_hidden_states
            residual_hidden_states = hidden_states

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )


class CompactSideT5Stack(T5PreTrainedModel):
    def __init__(self, config: MyT5Config, selected_layer_indices: List[int]):
        super().__init__(config)
        self.is_decoder = False
        self.selected_layer_indices = sorted(selected_layer_indices)
        self.block = nn.ModuleList(
            [
                T5Block(config, has_relative_attention_bias=bool(original_layer_idx == 0))
                for original_layer_idx in self.selected_layer_indices
            ]
        )
        self.final_layer_norm = T5LayerNorm(
            config.d_model, eps=config.layer_norm_epsilon
        )
        self.dropout = nn.Dropout(config.dropout_rate)
        self.gate_vectors = CompactGateWrapper(config, self.selected_layer_indices)
        self.gradient_checkpointing = False
        self.model_parallel = False

    def forward(
        self,
        inputs_embeds: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor],
        backbone_hidden_states: Tuple[torch.FloatTensor | None],
        output_hidden_states: Optional[bool] = None,
    ) -> BaseModelOutput:
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        input_shape = inputs_embeds.size()[:-1]
        batch_size, seq_length = input_shape
        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size, seq_length, device=inputs_embeds.device
            )
        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_shape
        )

        hidden_states = self.dropout(inputs_embeds)
        residual_hidden_states = hidden_states
        position_bias = None
        all_hidden_states = (hidden_states,) if output_hidden_states else None

        assert len(backbone_hidden_states) == self.config.num_layers
        for original_layer_idx, layer_module in zip(self.selected_layer_indices, self.block):
            selected_hidden = backbone_hidden_states[original_layer_idx]
            if selected_hidden is None:
                raise RuntimeError(
                    "EchoSlim compact denoiser expected selected layer "
                    f"{original_layer_idx}, but got None. Check lst_skip/server_layer_select."
                )

            gate = self.gate_vectors(original_layer_idx)
            hidden_states = selected_hidden * gate + hidden_states * (1 - gate)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                use_cache=False,
                output_attentions=False,
            )
            layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]
            hidden_states = layer_outputs[0]
            position_bias = layer_outputs[2]

            if self.config.use_residual:
                hidden_states = hidden_states + residual_hidden_states
            residual_hidden_states = hidden_states

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )


class Ladder(nn.Module):
    def __init__(
        self,
        config: MyT5Config,
        selected_layer_indices: Optional[List[int]] = None,
    ):
        super().__init__()
        self.config = config = deepcopy(config)
        self.selected_layer_indices = (
            None if selected_layer_indices is None else sorted(selected_layer_indices)
        )

        reduced_hidden_size = config.d_model // config.lst_reduce_factor
        self.emb_down = nn.Linear(config.d_model, reduced_hidden_size)

        reduced_config = deepcopy(config)
        reduced_config.d_model = reduced_hidden_size
        reduced_config.hidden_size = reduced_hidden_size
        reduced_config.d_ff = max(1, reduced_config.d_ff // config.lst_reduce_factor)
        if self.selected_layer_indices is None:
            self.trans = SideT5Stack(reduced_config)
        else:
            self.trans = CompactSideT5Stack(
                reduced_config,
                self.selected_layer_indices,
            )

        self.final_up = nn.Linear(reduced_hidden_size, config.d_model)
        if config.use_residual:
            nn.init.zeros_(self.final_up.weight)
            nn.init.zeros_(self.final_up.bias)

    def forward(
        self,
        input_embed,
        attention_mask,
        backbone_hidden_states: Tuple[torch.FloatTensor | None],
        output_hidden_states: Optional[bool] = None,
    ):
        input_embed = self.emb_down(input_embed)
        output = self.trans(
            inputs_embeds=input_embed,
            attention_mask=attention_mask,
            backbone_hidden_states=backbone_hidden_states,
            output_hidden_states=output_hidden_states,
        )
        up = self.final_up(output.last_hidden_state)
        return up, output.hidden_states if output_hidden_states else None


class ClientEmbeddingPart(nn.Module):
    def __init__(self, config: MyT5Config, embedding: nn.Embedding):
        super().__init__()
        self.config = config
        self.embed_tokens = embedding

    def forward(self, input_ids):
        clean_input_embeds = self.embed_tokens(input_ids)
        noisy_input_embeds, _ = get_noisy_embedding(
            clean_input_embeds,
            self.config.privacy_budget,
            clip=self.config.clip_embedding_l2,
            noise_type=self.config.noise_type,
            model_type="t5-large",
        )
        return clean_input_embeds, noisy_input_embeds


class ClientDenoisePart(nn.Module):
    def __init__(
        self,
        config: MyT5Config,
        selected_layer_indices: Optional[List[int]] = None,
    ):
        super().__init__()
        self.config = config
        self.selected_layer_indices = selected_layer_indices
        if config.lst_enable:
            self.ladder_side = Ladder(
                config,
                selected_layer_indices=selected_layer_indices,
            )

    def forward(
        self,
        all_hidden_states: list[torch.FloatTensor | None],
        attention_mask: torch.Tensor,
        clean_input_embeds: torch.FloatTensor | None = None,
        noisy_input_embeds: torch.FloatTensor | None = None,
        output_hidden_states: Optional[bool] = None,
    ):
        if self.config.lst_enable:
            if self.config.lst_input_type == "clean":
                emb = clean_input_embeds
            elif self.config.lst_input_type == "noisy":
                emb = noisy_input_embeds
            else:
                raise ValueError(f"Invalid lst_input_type: {self.config.lst_input_type}")
            return self.ladder_side(
                emb,
                attention_mask,
                all_hidden_states,
                output_hidden_states=output_hidden_states,
            )
        return all_hidden_states[-1], None


class SplitEncoderProxy(nn.Module):
    main_input_name = "input_ids"

    def __init__(self, parent_model: "SplittedT5ForConditionalGeneration"):
        super().__init__()
        self._parent_ref = weakref.ref(parent_model)
        self.config = parent_model.config

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
        **kwargs,
    ):
        parent_model = self._parent_ref()
        return parent_model.encode_with_noise(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class SplittedT5ForConditionalGeneration(T5PreTrainedModel):
    def __init__(self, config: MyT5Config, ptm_model: T5ForConditionalGeneration):
        super().__init__(config)
        self.model_dim = config.d_model
        self.shared = ptm_model.shared

        if (config.num_layers - 1) in config.lst_skip:
            config.lst_skip.remove(config.num_layers - 1)

        self.client_embedding = ClientEmbeddingPart(config, ptm_model.shared)
        self.server_encoder = T5EncoderWithStartLayer(config, ptm_model.encoder)
        self.encoder_proxy = SplitEncoderProxy(self)
        self.decoder = ptm_model.decoder
        self.lm_head = ptm_model.lm_head

        if config.lst_enable:
            self.server_layer_select = LayerSelect(config)
            self.server_downsample = HiddenDowns(config)
            self.selected_layer_indices: list[int] | None = None
            self.client_denoise = None
            defer_client_denoise = config.auto_skip and set(config.lst_skip or []) == {-1}
            if not defer_client_denoise:
                self._build_client_denoise_from_current_skip()

        self.post_init()

        self.total_embedding_data_transferred = 0
        self.total_hidden_states_data_transferred = 0

        self.mi_estimators = []
        self.mi_optimizers = []
        if config.lst_enable:
            if self.client_denoise is not None:
                self._build_mi_estimators(len(self.selected_layer_indices or []))

        print(f"Client params: {self._calc_client_params()}")

    def _selected_layers_from_config(self) -> list[int]:
        skip_layers = set(self.config.lst_skip or [])
        skip_layers.discard(-1)
        return [
            idx
            for idx in range(self.config.num_layers)
            if idx not in skip_layers
        ]

    def _build_client_denoise_from_current_skip(self, initialize_weights: bool = False):
        self.selected_layer_indices = self._selected_layers_from_config()
        self.client_denoise = ClientDenoisePart(
            self.config,
            selected_layer_indices=self.selected_layer_indices,
        )
        if initialize_weights:
            self.client_denoise.apply(self._init_weights)
        device = self.shared.weight.device
        dtype = self.shared.weight.dtype
        self.client_denoise.to(device=device, dtype=dtype)

    def _build_mi_estimators(self, num_reserved_layers: int):
        self.mi_estimators = []
        self.mi_optimizers = []
        if not self.config.mi_downsample_enable:
            return

        reduced_hidden_size = self.config.d_model // self.config.lst_reduce_factor
        self.mi_estimators = [
            (
                MINE(
                    self.config.d_model,
                    reduced_hidden_size,
                    self.config.mi_estimator_hidden_dim,
                ).to(self.device),
                MINE(
                    self.config.d_model,
                    reduced_hidden_size,
                    self.config.mi_estimator_hidden_dim,
                ).to(self.device),
            )
            for _ in range(num_reserved_layers)
        ]
        self.mi_optimizers = [
            (
                torch.optim.Adam(mi[0].parameters(), lr=self.config.mi_estimator_lr),
                torch.optim.Adam(mi[1].parameters(), lr=self.config.mi_estimator_lr),
            )
            for mi in self.mi_estimators
        ]

    def _init_client_denoise_from_encoder(self):
        if not self.config.lst_enable or self.client_denoise is None:
            return

        encoder_state_dict = self.server_encoder.state_dict()
        trans = self.client_denoise.ladder_side.trans
        selected_layer_indices = getattr(trans, "selected_layer_indices", None)

        for name, param in trans.named_parameters():
            if "gate_vectors" in name:
                nn.init.zeros_(param)
                continue
            if self.config.lst_random_init:
                continue

            source_name = name
            if "block" in name:
                compact_idx = int(re.search(r"block\.(\d+)", name).group(1))
                encoder_idx = (
                    selected_layer_indices[compact_idx]
                    if selected_layer_indices is not None
                    else compact_idx
                )
                source_name = re.sub(
                    r"block\.\d+", f"block.{encoder_idx}", source_name, count=1
                )

            if source_name not in encoder_state_dict:
                continue
            src = encoder_state_dict[source_name].data
            if len(param.shape) == 1:
                param.data.copy_(src[: param.shape[0]])
            elif len(param.shape) == 2:
                param.data.copy_(src[: param.shape[0], : param.shape[1]])
            else:
                param.data.copy_(src)

    def _accumulate_embedding_data_transferred(self, noisy_input_embeds):
        self.total_embedding_data_transferred += (
            noisy_input_embeds.numel() * noisy_input_embeds.element_size()
        )

    def _accumulate_hidden_states_data_transferred(self, all_hidden_states):
        for hidden_states in all_hidden_states:
            if hidden_states is None:
                continue
            self.total_hidden_states_data_transferred += (
                hidden_states.numel() * hidden_states.element_size()
            )

    def _calc_client_params(self):
        if not self.config.lst_enable:
            return 0
        if self.client_denoise is None:
            return 0
        return sum(p.numel() for p in self.client_denoise.parameters())

    @override
    def get_input_embeddings(self):
        return self.shared

    @override
    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.client_embedding.embed_tokens = new_embeddings
        self.server_encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    @override
    def get_encoder(self):
        return self.encoder_proxy

    @override
    def get_decoder(self):
        return self.decoder

    @override
    def get_output_embeddings(self):
        return self.lm_head

    @override
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def encode_with_noise(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
    ):
        if inputs_embeds is None:
            clean_input_embeds, noisy_input_embeds = self.client_embedding(input_ids)
            noise = (noisy_input_embeds - clean_input_embeds).detach()
        else:
            clean_input_embeds = noisy_input_embeds = inputs_embeds
            noise = None

        self._accumulate_embedding_data_transferred(noisy_input_embeds)

        encoder_outputs = self.server_encoder(
            None,
            attention_mask=attention_mask,
            inputs_embeds=noisy_input_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )

        all_hidden_states = encoder_outputs.hidden_states
        denoise_hidden_states = None
        all_ladder_hidden_states = None
        downsampled_hidden_states = None

        if self.config.lst_enable:
            if self.client_denoise is None:
                raise RuntimeError(
                    "EchoSlim client_denoise has not been built. "
                    "Call set_layer_skip() after HLF attribution before training/forward."
                )
            all_hidden_states = list(all_hidden_states[1:])
            all_hidden_states = self.server_layer_select(all_hidden_states)
            all_hidden_states = self.server_downsample(all_hidden_states)
            downsampled_hidden_states = all_hidden_states[:]

            self._accumulate_hidden_states_data_transferred(all_hidden_states)

            denoise_hidden_states, all_ladder_hidden_states = self.client_denoise(
                all_hidden_states,
                attention_mask,
                clean_input_embeds,
                noisy_input_embeds,
                output_hidden_states=output_hidden_states,
            )
            if self.config.use_residual:
                last_hidden_state = encoder_outputs.last_hidden_state + denoise_hidden_states
            else:
                last_hidden_state = denoise_hidden_states
        else:
            last_hidden_state = encoder_outputs.last_hidden_state

        output = BaseModelOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=encoder_outputs.hidden_states if output_hidden_states else None,
            attentions=encoder_outputs.attentions if output_attentions else None,
        )
        output.clean_input_embeds = clean_input_embeds
        output.noisy_input_embeds = noisy_input_embeds
        output.noise = noise
        output.downsampled_hidden_states = downsampled_hidden_states
        output.denoise_hidden_states = denoise_hidden_states
        output.all_ladder_hidden_states = all_ladder_hidden_states
        return output

    def _maybe_add_mi_loss(self, loss, encoder_outputs):
        if (
            loss is None
            or not self.config.lst_enable
            or not self.config.mi_downsample_enable
            or encoder_outputs.noise is None
        ):
            return loss

        filtered_downsampled_hidden_states = [
            hidden_state
            for hidden_state in encoder_outputs.downsampled_hidden_states
            if hidden_state is not None
        ]
        assert len(filtered_downsampled_hidden_states) == len(self.mi_estimators)

        layer_xz_mi_losses = []
        layer_yz_mi_losses = []
        for mi, mi_optimizer, down_hidden_state in zip(
            self.mi_estimators,
            self.mi_optimizers,
            filtered_downsampled_hidden_states,
        ):
            xz_mi, yz_mi = mi
            xz_mi_optimizer, yz_mi_optimizer = mi_optimizer

            if self.training:
                x_samples = encoder_outputs.noisy_input_embeds.detach()
                z_samples = down_hidden_state.detach()
                y_samples = encoder_outputs.denoise_hidden_states.detach()
                for _ in range(self.config.mi_estimator_iter_num):
                    xz_mi.train()
                    yz_mi.train()
                    xz_loss = xz_mi.learning_loss(x_samples, z_samples)
                    yz_loss = yz_mi.learning_loss(y_samples, z_samples)
                    xz_mi_optimizer.zero_grad()
                    yz_mi_optimizer.zero_grad()
                    xz_loss.backward()
                    yz_loss.backward()
                    xz_mi_optimizer.step()
                    yz_mi_optimizer.step()

            xz_mi.eval()
            yz_mi.eval()
            layer_xz_mi_losses.append(
                xz_mi(encoder_outputs.noisy_input_embeds, down_hidden_state)
            )
            layer_yz_mi_losses.append(
                yz_mi(encoder_outputs.denoise_hidden_states, down_hidden_state)
            )

        avg_xz_mi_loss = torch.stack(layer_xz_mi_losses).mean()
        avg_yz_mi_loss = torch.stack(layer_yz_mi_losses).mean()
        return loss + (
            avg_xz_mi_loss * self.config.mi_xz_ratio
            - avg_yz_mi_loss * self.config.mi_yz_ratio
        )

    @override
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encode_with_noise(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = self._shift_right(labels)

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]
        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            loss = self._maybe_add_mi_loss(loss, encoder_outputs)

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + (encoder_outputs,)
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        decoder_attention_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                remove_prefix_length = input_ids.shape[1] - 1
            input_ids = input_ids[:, remove_prefix_length:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def _reorder_cache(self, past_key_values, beam_idx):
        if past_key_values is None:
            return past_key_values
        reordered_decoder_past = ()
        for layer_past_states in past_key_values:
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )
            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past

    @classmethod
    def from_t5(cls, t5: T5ForConditionalGeneration):
        config = MyT5Config(**t5.config.to_dict())
        cls._set_default_torch_dtype(t5.dtype)
        model = cls(config, ptm_model=t5)

        if not model.config.lst_enable:
            return model

        model._init_client_denoise_from_encoder()
        return model

    @classmethod
    @override
    def from_pretrained(cls, *args, **kwargs):
        t5 = T5ForConditionalGeneration.from_pretrained(*args, **kwargs)
        return cls.from_t5(t5)

    def calc_layer_attributions(self, datasets: Iterable):
        layer_grads_sum = 0
        for batch in tqdm(datasets):
            device = self.client_embedding.embed_tokens.weight.device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            _, noisy_inputs_embeds = self.client_embedding(input_ids)

            encoder_outputs = self.server_encoder(
                None,
                attention_mask=attention_mask,
                inputs_embeds=noisy_inputs_embeds,
                output_hidden_states=True,
                return_dict=True,
            )
            batch_all_hidden_states = encoder_outputs.hidden_states

            layer_grads = []
            for idx, hidden_state in enumerate(batch_all_hidden_states[:-1]):
                integrated_grad = 0
                for step in range(1, self.config.num_integrate_step + 1):
                    scale_factor = step / self.config.num_integrate_step
                    scaled_hidden_state = (
                        scale_factor * hidden_state
                    ).detach().requires_grad_(True)
                    hidden = self.server_encoder(
                        None,
                        attention_mask=attention_mask,
                        inputs_embeds=scaled_hidden_state,
                        start_layer=idx + 1,
                        return_dict=True,
                    ).last_hidden_state
                    grad = torch.autograd.grad(hidden.mean(), scaled_hidden_state)[0]
                    integrated_grad += grad

                avg_grad = integrated_grad / self.config.num_integrate_step
                contribution = hidden_state * avg_grad
                valid_token_mask = attention_mask.unsqueeze(-1).to(
                    dtype=contribution.dtype,
                    device=contribution.device,
                )
                contribution = contribution * valid_token_mask
                layer_contribution = (
                    contribution.sum(dim=(0, 1))
                    / valid_token_mask.sum().clamp_min(1.0)
                ).unsqueeze(0)
                layer_grads.append(layer_contribution)

            layer_grads_sum += torch.stack(layer_grads)

        norms = torch.norm(layer_grads_sum, dim=(1, 2))
        topk = torch.topk(norms, self.config.num_layers)
        return topk.indices

    def set_layer_skip(
        self,
        sorted_layer_indices: torch.Tensor | List[int],
        num_reserved_layers: int,
        keep_last_layer: bool = False,
    ):
        if isinstance(sorted_layer_indices, torch.Tensor):
            sorted_layer_indices = sorted_layer_indices.tolist()
        sorted_layer_indices = sorted_layer_indices[:num_reserved_layers]
        selected_layers = set(sorted_layer_indices)
        if keep_last_layer:
            selected_layers.add(self.config.num_layers - 1)
        selected_layers = sorted(selected_layers)
        skip_layers = sorted(set(range(self.config.num_layers)) - set(selected_layers))

        self.config.lst_skip = skip_layers
        self.server_layer_select.lst_skip = self.config.lst_skip
        self._build_client_denoise_from_current_skip(initialize_weights=True)
        self._init_client_denoise_from_encoder()
        self._build_mi_estimators(len(self.selected_layer_indices or []))
        print(f"EchoSlim selected layers: {self.selected_layer_indices}")
        print(f"EchoSlim skipped layers: {self.config.lst_skip}")
