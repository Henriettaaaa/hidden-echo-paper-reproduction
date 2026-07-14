from copy import deepcopy
import re
from typing import Iterable, List, Optional, Tuple, Union
import numpy as np
from tqdm import tqdm
from typing_extensions import override
from torch import nn
import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import SequenceClassifierOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.cache_utils import Cache, StaticCache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast

from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2DecoderLayer,
    Qwen2ForSequenceClassification,
    Qwen2ForCausalLM,
    Qwen2MLP,
    Qwen2PreTrainedModel,
    Qwen2Model,
    Qwen2RMSNorm,
    logger,
)
from modeling.my.configuration import MyQwen2Config
from utils.noise import get_noisy_embedding, sample_noise_Chi

from sklearn.metrics import mutual_info_score

# HiddenEcho 的动态融合开关
# 控制“上一层 denoise 输出”和“当前 server hidden state”各占多少
class GateWrapper(nn.Module):
    def __init__(self, config: MyQwen2Config):
        super().__init__()
        self.config = config
        self.gate_vectors = nn.ParameterList( # 每一层一个可学习 gate 向量
            [
                nn.Parameter(torch.zeros(config.hidden_size))
                for _ in range(config.num_hidden_layers)
            ]
        )  

    def forward(self, layer_idx: int):
        return torch.sigmoid(self.gate_vectors[layer_idx] / self.config.lst_temperature)


# 客户端侧的轻量网络
class SideTransformerStack(nn.Module):
    def __init__(self, config: MyQwen2Config):
        super().__init__()
        self.config = config
        # side module 自己也有一套 transformer decoder layers
        self.dec_layers = nn.ModuleList(
            [
                Qwen2DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gate_vectors = GateWrapper(config) # 每一层有一个可学习 gate

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None, # 是否返回中间层
        *,
        backbone_hidden_states: Tuple[torch.FloatTensor], # server LLM 每层 hidden state
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )

        # use_cache = use_cache if use_cache is not None else self.config.use_cache
        use_cache = False

        assert inputs_embeds is not None

        use_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            use_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
                "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
            )

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values,
            output_attentions,
        )

        # 先用本地 embedding 初始化 side 分支
        hidden_states = inputs_embeds

        # decoder layers
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        assert len(backbone_hidden_states) == self.config.num_hidden_layers
        # 期望 server 返回的 hidden states 数量和 LLM 层数一致。
        # 如果 HiddenEcho+ 跳过某些层，那些位置会是 None
        
        all_hidden_states = None
        if output_hidden_states:
            all_hidden_states = (hidden_states,)

        residual_hidden_states = hidden_states
        # HiddenEcho+ 的 layer selection：如果某层没传回来，就跳过
        for layer_idx, decoder_layer in enumerate(self.dec_layers):
            if backbone_hidden_states[layer_idx] is None:
                continue

            # gate = torch.sigmoid(
            #     self.gate_vectors[layer_idx] / self.config.lst_temperature
            # )
            gate = self.gate_vectors(layer_idx)
            
            # 混合 H_i 和当前 side hidden
            hidden_states = backbone_hidden_states[layer_idx] * gate + hidden_states * (
                1 - gate
            )
            
            # 过 side transformer 的第 i 层
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

            hidden_states = layer_outputs[0] # 从 Qwen2DecoderLayer 的输出取第一项，就是整个隐藏层
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.config.use_residual:
                hidden_states = hidden_states + residual_hidden_states
            residual_hidden_states = hidden_states

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache()
                if use_legacy_cache
                else next_decoder_cache
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states, # 复现分类任务
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    # Copied from transformers.models.llama.modeling_llama.LlamaModel._update_causal_mask
    # 因果注意力掩码
    # 让 side transformer 的 attention 行为和原模型一致
    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
        # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
        # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
        # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114

        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if (
            self.config._attn_implementation == "sdpa"
            and not using_static_cache
            and not output_attentions
        ):
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        if attention_mask is not None and attention_mask.dim() == 4:
            # in this case we assume that the mask comes already in inverted form and requires no inversion or slicing
            if attention_mask.max() != 0:
                raise ValueError(
                    "Custom 4D attention mask should be passed in inverted form with max==0`"
                )
            causal_mask = attention_mask
        else:
            causal_mask = torch.full(
                (sequence_length, target_length),
                fill_value=min_dtype,
                dtype=dtype,
                device=device,
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(
                target_length, device=device
            ) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(
                input_tensor.shape[0], 1, -1, -1
            )
            if attention_mask is not None:
                causal_mask = (
                    causal_mask.clone()
                )  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = (
                    causal_mask[:, :, :, :mask_length]
                    + attention_mask[:, None, None, :]
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[
                    :, :, :, :mask_length
                ].masked_fill(padding_mask, min_dtype)
        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(
                causal_mask, min_dtype
            )

        return causal_mask


# EchoSlim 的 compact gate。
# 原 HiddenEcho+ 虽然 forward 跳过未选层，但 GateWrapper 仍为全部 L 层建参数。
# 这里每个 gate 只对应一个真正被 HLF 选中的原始层号，避免 client 端保留无用 gate。
class CompactGateWrapper(nn.Module):
    def __init__(self, config: MyQwen2Config, selected_layer_indices: List[int]):
        super().__init__()
        self.config = config
        self.selected_layer_indices = list(selected_layer_indices)
        self.layer_to_compact_idx = {
            layer_idx: compact_idx
            for compact_idx, layer_idx in enumerate(self.selected_layer_indices)
        }
        self.gate_vectors = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(config.hidden_size))
                for _ in self.selected_layer_indices
            ]
        )

    def forward(self, original_layer_idx: int):
        compact_idx = self.layer_to_compact_idx[original_layer_idx]
        return torch.sigmoid(
            self.gate_vectors[compact_idx] / self.config.lst_temperature
        )


# EchoSlim 的结构稀疏 side stack。
# 与原 SideTransformerStack 不同，这里只实例化 HLF 选中的 k 个 side layers；
# 但每个 block 仍记录自己的“原始层号”，用于 gate 对齐和从 backbone 对应层初始化。
class CompactSideTransformerStack(nn.Module):
    def __init__(self, config: MyQwen2Config, selected_layer_indices: List[int]):
        super().__init__()
        self.config = config
        self.selected_layer_indices = sorted(selected_layer_indices)
        self.dec_layers = nn.ModuleList(
            [
                Qwen2DecoderLayer(config, original_layer_idx)
                for original_layer_idx in self.selected_layer_indices
            ]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gate_vectors = CompactGateWrapper(config, self.selected_layer_indices)

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        *,
        backbone_hidden_states: Tuple[torch.FloatTensor],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        use_cache = False

        assert inputs_embeds is not None
        assert len(backbone_hidden_states) == self.config.num_hidden_layers

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values,
            output_attentions,
        )

        hidden_states = inputs_embeds
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        all_hidden_states = (hidden_states,) if output_hidden_states else None
        residual_hidden_states = hidden_states

        # 只遍历被选中的 compact blocks。backbone_hidden_states 仍保持 L 长度，
        # 这样 server 选层、通信统计、MI loss 和 client denoise 的层号都一致。
        for original_layer_idx, decoder_layer in zip(
            self.selected_layer_indices, self.dec_layers
        ):
            selected_hidden = backbone_hidden_states[original_layer_idx]
            if selected_hidden is None:
                raise RuntimeError(
                    "EchoSlim compact denoiser expected selected layer "
                    f"{original_layer_idx}, but got None. Check lst_skip/server_layer_select."
                )

            gate = self.gate_vectors(original_layer_idx)
            hidden_states = selected_hidden * gate + hidden_states * (1 - gate)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )
            hidden_states = layer_outputs[0]
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.config.use_residual:
                hidden_states = hidden_states + residual_hidden_states
            residual_hidden_states = hidden_states

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache()
                if next_decoder_cache is not None
                else None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    # 与原 side stack 保持一致的 causal mask 逻辑，避免 compact 后 attention 行为变化。
    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        return SideTransformerStack._update_causal_mask(
            self,
            attention_mask,
            input_tensor,
            cache_position,
            past_key_values,
            output_attentions,
        )

# HiddenEcho+ 的压缩层。
# 对每个 hidden state 做线性下采样，把维度从 d 压到 d / r
class HiddenDowns(nn.Module):
    def __init__(self, config: MyQwen2Config):
        super().__init__()
        self.config = config

        if config.lst_reduce_factor == 1:
            self.hidden_downs = nn.ModuleList(
                [nn.Identity() for _ in range(config.num_hidden_layers)]
            )  # pass through
        else:
            reduced_hidden_size = config.hidden_size // config.lst_reduce_factor
            self.hidden_downs = nn.ModuleList(
                [
                    nn.Linear(config.hidden_size, reduced_hidden_size)
                    for _ in range(config.num_hidden_layers)
                ]
            )

    def forward(
        self, hidden_states: List[torch.FloatTensor | None]
    ) -> List[torch.FloatTensor | None]:
        assert len(hidden_states) == self.config.num_hidden_layers
        return [
            down(hidden_state) if hidden_state is not None else None
            for down, hidden_state in zip(self.hidden_downs, hidden_states)
        ]

# 完整的 denoise module（去噪模块）
# 降维 -> SideTransformerStack -> 升维
class Ladder(nn.Module):
    def __init__(
        self,
        config: MyQwen2Config,
        selected_layer_indices: Optional[List[int]] = None,
    ):
        super().__init__()
        self.config = config = deepcopy(config)
        self.selected_layer_indices = (
            None if selected_layer_indices is None else sorted(selected_layer_indices)
        )

        reduced_hidden_size = config.hidden_size // config.lst_reduce_factor
        self.emb_down = nn.Linear(config.hidden_size, reduced_hidden_size)

        reduced_config = deepcopy(config)
        reduced_config.hidden_size = reduced_hidden_size
        reduced_config.intermediate_size = (
            reduced_config.intermediate_size // config.lst_reduce_factor
        )
        # EchoSlim：如果传入 selected_layer_indices，只实例化被 HLF 选中的 side layers。
        # 未传入时退化为原 HiddenEcho/HiddenEcho+ 的完整 side stack，方便非 auto_skip 配置对照。
        if self.selected_layer_indices is None:
            self.trans = SideTransformerStack(reduced_config)
        else:
            self.trans = CompactSideTransformerStack(
                reduced_config, self.selected_layer_indices
            )

        self.final_up = nn.Linear(reduced_hidden_size, config.hidden_size)

    def forward(
        self,
        input_embed,
        attention_mask,
        backbone_hidden_states: Tuple[torch.FloatTensor],
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
        return (
            up,
            output.hidden_states if output_hidden_states else None,
        )


class ClientEmbeddingPart(nn.Module):
    def __init__(self, config: MyQwen2Config, embedding: nn.Embedding):
        super().__init__()
        self.config = config
        self.embed_tokens = embedding

    def forward(self, input_ids):
        inputs_embeds = self.embed_tokens(input_ids)
        clean_input_embeds = inputs_embeds # 客户端词嵌入，加噪前
        noisy_input_embeds, noise = get_noisy_embedding(
            clean_input_embeds,
            self.config.privacy_budget,
            clip=self.config.clip_embedding_l2,
            noise_type=self.config.noise_type,
        ) # 这里的噪声默认是 Chi 噪声，不是 Gaussian，在config里定义
        return clean_input_embeds, noisy_input_embeds

# 客户端去噪入口
# 是否启用 denoiser？
# 用 clean embedding 还是 noisy embedding？
# 调用真正的 Ladder。
class ClientDenoisePart(nn.Module):
    def __init__(
        self,
        config: MyQwen2Config,
        selected_layer_indices: Optional[List[int]] = None,
    ):
        super().__init__()
        self.config = config
        self.selected_layer_indices = selected_layer_indices
        if config.lst_enable:
            self.ladder_side = Ladder(
                config,
                selected_layer_indices=selected_layer_indices,
            ) # 包住完整或 EchoSlim compact 的 denoise module（去噪模块）

    def forward(
        self,
        all_hidden_states: list[torch.FloatTensor],
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
                raise ValueError(
                    f"Invalid lst_input_type: {self.config.lst_input_type}"
                )
            hidden_states, all_ladder_hidden_states = self.ladder_side(
                emb,
                attention_mask,
                all_hidden_states,
                output_hidden_states=output_hidden_states,
            )
            return hidden_states, all_ladder_hidden_states
        else:
            return all_hidden_states[-1], None

# 分类头。拿去噪后的 hidden state 做线性分类，按最后一个非 padding token 聚合，计算交叉熵 / 回归损失。
# 这和 HF 的 Qwen2ForSequenceClassification 逻辑一致。
class ClientHeadPart(nn.Module):
    def __init__(self, config: MyQwen2Config, head: nn.Linear):
        super().__init__()
        self.config = config
        self.num_labels = config.num_labels
        self.score = head

    def forward(
        self,
        input_ids: torch.LongTensor,
        hidden_states: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
    ):
        logits = self.score(hidden_states)

        batch_size = hidden_states.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError(
                "Cannot handle batch sizes > 1 if no padding token is defined."
            )
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = (
                    torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                )
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[
            torch.arange(batch_size, device=logits.device), sequence_lengths
        ]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    pooled_logits.view(-1, self.num_labels), labels.view(-1)
                )
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)

        return loss, pooled_logits



# 语言模型头，给生成任务留的接口
# TODO: 似乎可能没写全
class ClientLmHeadPart(nn.Module):
    def __init__(self, config: MyQwen2Config, head: nn.Linear):
        super().__init__()
        self.config = config
        self.lm_head = head

    def forward(
        self,
        input_ids: torch.LongTensor,
        hidden_states: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
    ):
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return loss, logits



# 对 HuggingFace Qwen2Model 的包装，以支持 start_layer，允许从中间某一层开始跑
class MyQwen2Model(Qwen2PreTrainedModel):

    def __init__(self, config: MyQwen2Config, model: Qwen2Model):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = model.embed_tokens
        self.layers = model.layers
        self._attn_implementation = config._attn_implementation
        self.norm = model.norm

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        *,
        start_layer: int = 0, # 改动（新增参数）
    ) -> Union[Tuple, BaseModelOutputWithPast]:
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
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        use_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            use_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
                "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
            )

        # 支持 inputs_embeds=scaled_hidden_state
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) # 没指定层数时，默认跑全部层

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values,
            output_attentions,
        )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        
        for decoder_layer in self.layers[start_layer:]: # 改动
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache()
                if use_legacy_cache
                else next_decoder_cache
            )

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    # Copied from transformers.models.llama.modeling_llama.LlamaModel._update_causal_mask
    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
        # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
        # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
        # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114

        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if (
            self.config._attn_implementation == "sdpa"
            and not using_static_cache
            and not output_attentions
        ):
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        if attention_mask is not None and attention_mask.dim() == 4:
            # in this case we assume that the mask comes already in inverted form and requires no inversion or slicing
            if attention_mask.max() != 0:
                raise ValueError(
                    "Custom 4D attention mask should be passed in inverted form with max==0`"
                )
            causal_mask = attention_mask
        else:
            causal_mask = torch.full(
                (sequence_length, target_length),
                fill_value=min_dtype,
                dtype=dtype,
                device=device,
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(
                target_length, device=device
            ) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(
                input_tensor.shape[0], 1, -1, -1
            )
            if attention_mask is not None:
                causal_mask = (
                    causal_mask.clone()
                )  
                mask_length = attention_mask.shape[-1]
                padding_mask = (
                    causal_mask[:, :, :, :mask_length]
                    + attention_mask[:, None, None, :]
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[
                    :, :, :, :mask_length
                ].masked_fill(padding_mask, min_dtype)
        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(
                causal_mask, min_dtype
            )

        return causal_mask

# HiddenEcho+ 的层过滤器。只传重要层
class LayerSelect(nn.Module):
    def __init__(self, config: MyQwen2Config):
        super().__init__()
        self.config = config
        self.lst_skip = config.lst_skip

    def forward(self, hidden_states: List[torch.FloatTensor]):
        assert len(hidden_states) == self.config.num_hidden_layers
        return [
            hidden_state if idx not in self.lst_skip else None
            for idx, hidden_state in enumerate(hidden_states)
        ]


# 互信息估计器，是HiddenEcho+ 的约束项
# 只在：mi_downsample_enable=True 时参与训练。
# copy from https://github.com/Linear95/CLUB/blob/master/mi_estimators.py
# 对应论文里的 information bottleneck（信息瓶颈）：
# 希望 downsample 后的 hidden state：
# 1. 少保留 noisy embedding 里的噪声信息 I(noisy_input_embeds, down_hidden_state)
# 2. 多保留 denoised output 需要的任务信息 I(denoised_hidden_states, down_hidden_state)
# 用两个 MINE 网络分别近似
# 最终加到 loss
class MINE(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size):
        super(MINE, self).__init__()
        self.T_func = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )

    def forward(self, x_samples, y_samples):  # samples have shape [sample_size, dim]
        # shuffle and concatenate
        sample_size = y_samples.shape[0]
        random_index = torch.randint(sample_size, (sample_size,)).long()

        y_shuffle = y_samples[random_index]

        # MINE 输入拼接后的向量，维度 d + d'
        T0 = self.T_func(torch.cat([x_samples, y_samples], dim=-1)) # 真实配对的 (x, y)--联合分布
        T1 = self.T_func(torch.cat([x_samples, y_shuffle], dim=-1)) # 打乱配对--边缘分布

        # 互信息下界估计（可训练），越大越好
        T1_flat = T1.float().reshape(-1)
        log_mean_exp = torch.logsumexp(T1_flat, dim=0) - T1_flat.new_tensor(T1_flat.numel()).log()
        lower_bound = T0.float().mean() - log_mean_exp

        # compute the negative loss (maximise loss == minimise -loss)
        return lower_bound

    def learning_loss(self, x_samples, y_samples):
        return -self.forward(x_samples, y_samples) #loss本身是越小越好，所以对 lower_bound 取负数，作为目标


class SplittedQwen2ForSequenceClassification(Qwen2PreTrainedModel):
    def __init__(
        self, config: MyQwen2Config, ptm_model: Qwen2ForSequenceClassification | Qwen2ForCausalLM
    ):
        super().__init__(config)

        head_cls = ClientHeadPart
        ptm_model_head = ptm_model.score # 取原模型的分类头

        
        if (config.num_hidden_layers - 1) in config.lst_skip:
            config.lst_skip.remove(config.num_hidden_layers - 1)

        self.client_embedding = ClientEmbeddingPart(
            config, ptm_model.get_input_embeddings()
        )

        self.server_backbone = MyQwen2Model(config, ptm_model.model)
        if config.lst_enable: # 开echo
            self.server_layer_select = LayerSelect(config)
            self.server_downsample = HiddenDowns(config)

            self.selected_layer_indices: list[int] | None = None
            # EchoSlim 的关键改动：
            # auto_skip=True 时，HLF 还没计算出最终 selected layers，
            # 因此这里先不实例化完整 L 层 client_denoise，避免训练前就持有冗余 side layers。
            # train_split_echoslim.py 会先调用 calc_layer_attributions/set_layer_skip，
            # set_layer_skip 内部再根据最终 selected layers 构建 compact denoiser。
            self.client_denoise = None
            defer_client_denoise = config.auto_skip and set(config.lst_skip or []) == {-1}
            if not defer_client_denoise:
                self._build_client_denoise_from_current_skip()
            self.client_head = head_cls(config, ptm_model_head)
        else: # 不加去噪模块
            self.server_head = head_cls(config, ptm_model_head)

        self.post_init()

        self.total_embedding_data_transferred = 0
        self.total_hidden_states_data_transferred = 0

        self.mi_estimators = []
        self.mi_optimizers = []
        if config.lst_enable and self.client_denoise is not None:
            self._build_mi_estimators(len(self.selected_layer_indices or []))

        print(f"Client params: {self._calc_client_params()}")

    def _selected_layers_from_config(self) -> list[int]:
        skip_layers = set(self.config.lst_skip or [])
        skip_layers.discard(-1)
        return [
            idx
            for idx in range(self.config.num_hidden_layers)
            if idx not in skip_layers
        ]

    def _build_client_denoise_from_current_skip(self, initialize_weights: bool = False):
        self.selected_layer_indices = self._selected_layers_from_config()
        self.client_denoise = ClientDenoisePart(
            self.config,
            selected_layer_indices=self.selected_layer_indices,
        )
        if initialize_weights:
            # 延迟构建发生在 post_init() 之后，需要显式执行 HF 初始化，
            # 使 emb_down/final_up 等新增参数和原始构建路径保持一致。
            self.client_denoise.apply(self._init_weights)
        device = self.server_backbone.embed_tokens.weight.device
        dtype = self.server_backbone.embed_tokens.weight.dtype
        self.client_denoise.to(device=device, dtype=dtype)

    def _build_mi_estimators(self, num_reserved_layers: int):
        # MI estimator 的数量必须和实际被回传/被 compact denoiser 消费的层数一致。
        # 原实现对 auto_skip 预估 k+1，EchoSlim 这里在 selected layers 已确定后精确创建。
        self.mi_estimators = []
        self.mi_optimizers = []
        if not self.config.mi_downsample_enable:
            return

        reduced_hidden_size = self.config.hidden_size // self.config.lst_reduce_factor
        self.mi_estimators = [
            (
                MINE(
                    self.config.hidden_size,
                    reduced_hidden_size,
                    self.config.mi_estimator_hidden_dim,
                ).to(self.device),
                MINE(
                    self.config.hidden_size,
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

    def _init_client_denoise_from_backbone(self):
        if not self.config.lst_enable or self.client_denoise is None:
            return

        backbone_state_dict = self.server_backbone.state_dict()
        trans = self.client_denoise.ladder_side.trans
        selected_layer_indices = getattr(trans, "selected_layer_indices", None)

        for name, param in trans.named_parameters():
            if "gate_vectors" in name:
                nn.init.zeros_(param)
                continue
            if self.config.lst_random_init:
                continue

            source_name = name
            if "dec_layers" in name:
                compact_idx = int(re.search(r"dec_layers\.(\d+)", name).group(1))
                # EchoSlim 初始化必须按原始层号对齐：
                # compact 第 compact_idx 个 block 对应 selected_layer_indices[compact_idx]，
                # 不能按 compact_idx 直接拷 backbone 的同序号层。
                backbone_idx = (
                    selected_layer_indices[compact_idx]
                    if selected_layer_indices is not None
                    else compact_idx
                )
                source_name = name.replace("dec_layers", "layers")
                source_name = re.sub(
                    r"layers\.\d+", f"layers.{backbone_idx}", source_name, count=1
                )

            if source_name not in backbone_state_dict:
                continue
            source_param = backbone_state_dict[source_name].data
            if len(param.shape) == 1:
                param.data.copy_(source_param[: param.shape[0]])
            else:
                param.data.copy_(
                    source_param[: param.shape[0], : param.shape[1]]
                )

    # 统计 embedding 上传了多少字节
    def _accumulate_embedding_data_transferred(
        self, noisy_input_embeds: torch.FloatTensor
    ):
        self.total_embedding_data_transferred += (
            noisy_input_embeds.numel() * noisy_input_embeds.element_size()
        )
    # 统计 server 返回给 client 的 hidden states 通信量
    def _accumulate_hidden_states_data_transferred(
        self, all_hidden_states: list[torch.FloatTensor]
    ):
        for hidden_states in all_hidden_states:
            if hidden_states is None:
                continue
            self.total_hidden_states_data_transferred += (
                hidden_states.numel() * hidden_states.element_size()
            )
    # 计算端侧参数量
    def _calc_client_params(self):
        if not self.config.lst_enable:
            return 0
        denoise_params = (
            0
            if self.client_denoise is None
            else sum(p.numel() for p in self.client_denoise.parameters())
        )
        return denoise_params + sum(
            p.numel() for p in self.client_head.parameters()
        )

    # 返回输入 embedding 层
    @override
    def get_input_embeddings(self):
        return self.client_embedding.embed_tokens

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        # 嵌入层：加噪/不加噪
        if inputs_embeds is None:
            clean_input_embeds, noisy_input_embeds = self.client_embedding(input_ids)
            noise = (noisy_input_embeds - clean_input_embeds).detach()
            # clean_input_embeds, noisy_input_embeds, noise_loss = self.client_embedding(input_ids)
        else:
            clean_input_embeds = noisy_input_embeds = inputs_embeds
            noise = None

        
        self._accumulate_embedding_data_transferred(noisy_input_embeds)
        # 经过服务器的transformer堆叠
        transformer_outputs = self.server_backbone(
            None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=noisy_input_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        # 取出 server 所有 hidden states
        all_hidden_states = transformer_outputs.hidden_states
        if not output_hidden_states:
            transformer_outputs.hidden_states = None

        if self.config.lst_enable:
            if self.client_denoise is None:
                raise RuntimeError(
                    "EchoSlim client_denoise has not been built. "
                    "Call set_layer_skip() after HLF attribution before training/forward."
                )
            all_hidden_states = all_hidden_states[1:]  
            # HiddenEcho+ 层选择、降维
            all_hidden_states = self.server_layer_select(all_hidden_states)
            all_hidden_states = self.server_downsample(all_hidden_states)

            downsampled_hidden_states = all_hidden_states[:]

            
            self._accumulate_hidden_states_data_transferred(all_hidden_states)
            # 客户端去噪
            hidden_states, all_ladder_hidden_states = self.client_denoise(
                all_hidden_states,
                attention_mask,
                clean_input_embeds,
                noisy_input_embeds,
                output_hidden_states=output_hidden_states,
            )
            # 任务头算分类 loss
            loss, pooled_logits = self.client_head(input_ids, hidden_states, labels)
            # 如果启用 MI 正则，额外加 loss
            if loss is not None and self.config.mi_downsample_enable and noise is not None:
                filtered_downsampled_hidden_states = [
                    hidden_state
                    for hidden_state in downsampled_hidden_states
                    if hidden_state is not None
                ]

                assert len(filtered_downsampled_hidden_states) == len(
                    self.mi_estimators
                )

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
                        x_samples = noisy_input_embeds.detach()
                        
                        z_samples = down_hidden_state.detach()
                        y_samples = hidden_states.detach()
                        avg_xz_mi_loss = 0
                        avg_yz_mi_loss = 0
                        ITER_NUM = self.config.mi_estimator_iter_num
                        for _ in range(ITER_NUM):
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
                            avg_xz_mi_loss += xz_loss.item()
                            avg_yz_mi_loss += yz_loss.item()
                        avg_xz_mi_loss /= ITER_NUM
                        avg_yz_mi_loss /= ITER_NUM
                        # print(
                        #     f"train -- avg_xz_mi_loss: {avg_xz_mi_loss}, avg_yz_mi_loss: {avg_yz_mi_loss}"
                        # )

                    x_samples = noisy_input_embeds
                    
                    z_samples = down_hidden_state
                    y_samples = hidden_states

                    xz_mi.eval()
                    yz_mi.eval()
                    xz_loss = xz_mi(x_samples, z_samples)
                    yz_loss = yz_mi(y_samples, z_samples)
                    layer_xz_mi_losses.append(xz_loss)
                    layer_yz_mi_losses.append(yz_loss)

                # print(
                #     f"xz_mi: {[round(loss.item(), 3) for loss in layer_xz_mi_losses]}"
                # )
                # print(
                #     f"yz_mi: {[round(loss.item(), 3) for loss in layer_yz_mi_losses]}"
                # )

                avg_xz_mi_loss = torch.stack(layer_xz_mi_losses).mean()
                avg_yz_mi_loss = torch.stack(layer_yz_mi_losses).mean()
                # print(
                #     f"avg_xz_mi_loss: {avg_xz_mi_loss}, avg_yz_mi_loss: {avg_yz_mi_loss}"
                # )
                loss += (
                    avg_xz_mi_loss * self.config.mi_xz_ratio
                    - avg_yz_mi_loss * self.config.mi_yz_ratio
                )

        else:
            loss, pooled_logits = self.server_head(
                input_ids, all_hidden_states[-1], labels
            )

        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        output = SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
        if self.config.lst_enable and output_hidden_states:
            output.all_ladder_hidden_states = all_ladder_hidden_states
            output.downsampled_hidden_states = downsampled_hidden_states
            output.denoise_hidden_states = hidden_states
        return output

    # 把预训练模型参数拷到新模块里时做 shape 对齐截断
    @classmethod
    def _init_by_ptm(cls, param_to_init: nn.Parameter, ptm_param: nn.Parameter):
        if len(param_to_init.shape) == 1:
            param_to_init.data = ptm_param.data[: param_to_init.shape[0]]
        elif len(param_to_init.shape) == 2:
            param_to_init.data = ptm_param.data[
                : param_to_init.shape[0], : param_to_init.shape[1]
            ]
        else:
            raise ValueError(f"Invalid shape: {param_to_init.shape}")
    # 把原 Qwen2 的权重迁移到 split / denoise 结构里
    @classmethod
    def from_qwen2(cls, qwen2: Qwen2ForSequenceClassification | Qwen2ForCausalLM):
        config = deepcopy(qwen2.config)
        cls._set_default_torch_dtype(qwen2.dtype)

        model = cls(config, ptm_model=qwen2)

        if not model.config.lst_enable:
            return model

        if (model.config.num_hidden_layers - 1) in model.config.lst_skip:
            model.config.lst_skip.remove(model.config.num_hidden_layers - 1)

        # 非 auto_skip 时 denoiser 已在 __init__ 中构建；auto_skip 时会在 set_layer_skip()
        # 拿到 HLF 结果后构建，并在那里再次调用该初始化逻辑。
        model._init_client_denoise_from_backbone()

        return model
    # 先载入原 Qwen2，再 调用 from_qwen2(qwen2) 改造成 split 版
    @classmethod
    @override
    def from_pretrained(
        cls,
        *args,
        **kwargs,
    ):
        cls_type = Qwen2ForSequenceClassification
        qwen2 = cls_type.from_pretrained(
            *args,
            **kwargs,
        )
        return cls.from_qwen2(qwen2)
    
    # 算每一层 hidden state 的重要性，用于 HiddenEcho+ 的层选择
    def calc_layer_attributions(self, datasets: Iterable):
        
        layer_grads_sum = 0  

        for batch in tqdm(datasets):
            device = self.client_embedding.embed_tokens.weight.device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            clean_inputs_embeds, noisy_inputs_embeds = self.client_embedding(input_ids)

            transformer_outputs = self.server_backbone(
                None,
                attention_mask=attention_mask,
                inputs_embeds=noisy_inputs_embeds,
                output_hidden_states=True,
            )
            batch_all_hidden_states = transformer_outputs.hidden_states

            layer_grads = []  
            for idx, hidden_state in enumerate(batch_all_hidden_states[:-1]):
                hidden_state.requires_grad_()

                # # HiddenEcho+ 的梯度积分层选择（因为原代码与论文公式不符，所以重新写）
                # integrated_grad = 0  
                # STEPS = self.config.num_integrate_step
                # for scale_factor in torch.linspace(
                #     0, 1, STEPS, device=hidden_state.device, dtype=hidden_state.dtype
                # ):
                #     scaled_hidden_state = scale_factor * hidden_state
                #     # 不是 token embedding，而是某一层的 hidden state
                #     # 本质上是“当前层输入表示”

                #     hidden = self.server_backbone(
                #         None,
                #         attention_mask=attention_mask,
                #         inputs_embeds=scaled_hidden_state,
                #         use_cache=False,
                #         start_layer=idx + 1,
                #         # hidden_state 是第 idx 层输出附近的表示。
                #         # 要看它对最终输出的影响，就应该从下一层继续跑，所以+1
                #     ).last_hidden_state

                    
                #     grad = torch.autograd.grad(
                #         hidden.mean(),
                #         scaled_hidden_state,
                #     )[0]
                #     grad_sum = grad.sum(dim=0)  
                #     integrated_grad += grad_sum  
                # integrated_grad = integrated_grad * (1 / (STEPS - 1))  
                # layer_grads.append(integrated_grad)

                # TODO:HiddenEcho+ 的梯度积分层选择 修正，以及可能的修改
                # Σ_j ∂H_{L-1} / ∂H_i |_{H_i = (j/m)H_i}
                # 仍然用 hidden.mean() 作为 scalarization。更接近公式，但不是完整 Jacobian。
                # 如果要 task-aware，则目标应改成 task logit 或 loss，例如分类任务的 gold-label logit / loss      
                
                integrated_grad = 0 # 累计梯度求和项
                STEPS = self.config.num_integrate_step

                for step in range(1, STEPS + 1):
                    # 论文公式 j = 1 ... m，插值点是 (j/m) * H_i。
                    # 不包含 0，包含原始 hidden_state。
                    scale_factor = step / STEPS
                    
                    # detach()，切断它和前面 forward graph 的关系，避免反复回传到更早层；让每个积分点成为独立求梯度的输入。
                    # requires_grad_(True)，计算最终 hidden 对当前 H_hat_i 的梯度。
                    scaled_hidden_state = ( # 当前积分点上的 hidden state
                        scale_factor * hidden_state
                    ).detach().requires_grad_(True)

                    # hidden_states是第 idx 层输出，
                    # 要估计它对最终 hidden 的影响，需要把它作为下一层输入。
                    # 从第 idx + 1 层继续 forward
                    hidden = self.server_backbone(
                        None,
                        attention_mask=attention_mask,
                        inputs_embeds=scaled_hidden_state,
                        use_cache=False,
                        start_layer=idx + 1,
                    ).last_hidden_state

                    # 将最终 hidden state 标量化。
                    # 这。
                    # 如果要做 task-aware attribution，应把这里换成 task logit 或 loss。
                    # 返回形状与 scaled_hidden_state 相同：[batch_size, seq_len, hidden_dim]
                    grad = torch.autograd.grad(
                        hidden.mean(),
                        scaled_hidden_state,
                    )[0]
                    
                    integrated_grad += grad # 累计所有积分点的梯度（沿路径的平均敏感度）

                avg_grad = integrated_grad / STEPS

                #  H_i 乘子： 该层实际激活值
                contribution = hidden_state * avg_grad

                # 如果存在 padding，需要避免 padding token 参与层贡献统计。
                # attention_mask:[batch_size, seq_len]
                # valid_token_mask:[batch_size, seq_len, 1]
                valid_token_mask = attention_mask.unsqueeze(-1).to(
                    dtype=contribution.dtype,
                    device=contribution.device,
                )

                contribution = contribution * valid_token_mask

                # 对 batch 维度取平均。
                # 输出形状：[seq_len, hidden_dim]
                layer_contribution = contribution.sum(dim=0) / valid_token_mask.sum(
                    dim=0
                ).clamp_min(1.0)

                layer_grads.append(layer_contribution)

            layer_grads_sum += torch.stack(layer_grads)

        norms = torch.norm(layer_grads_sum, dim=(1, 2))  
        topk = torch.topk(norms, self.config.num_hidden_layers)
        return topk.indices
    # 把“要跳过哪些层”写进配置里
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
        # 是否强制保留最后一层
        if keep_last_layer:
            selected_layers.add(self.config.num_hidden_layers - 1)
        selected_layers = sorted(selected_layers)
        skip_layers = sorted(
            set(range(self.config.num_hidden_layers)) - set(selected_layers)
        )

        self.config.lst_skip = skip_layers
        self.server_layer_select.lst_skip = skip_layers

        # EchoSlim：HLF mask 一旦确定，立即把同一组 selected layers 固化到 client 结构中。
        # 这里同时重建 compact denoiser、按原始层号初始化权重、并让 MI estimator 数量对齐。
        self._build_client_denoise_from_current_skip(initialize_weights=True)
        self._init_client_denoise_from_backbone()
        self._build_mi_estimators(len(self.selected_layer_indices or []))
        print(f"EchoSlim selected layers: {self.selected_layer_indices}")
        print(f"EchoSlim skipped layers: {self.config.lst_skip}")
