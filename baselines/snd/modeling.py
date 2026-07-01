
import math
from typing import List, Optional, Tuple, Union

import torch
import os
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model, Qwen2PreTrainedModel
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaModel, LlamaPreTrainedModel

from utils.noise import get_noisy_embedding


def _clip_embedding_l2_enabled():
    value = os.environ.get("SND_CLIP_EMBEDDING_L2", "true").lower()
    return value in {"1", "true", "yes", "y"}


class DenoiseModel(Qwen2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model(config)

        
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        noisy_cls_embed: torch.FloatTensor,
        noise: torch.FloatTensor,
        clean_cls_embed: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        
        return_dict = self.config.use_return_dict
        
        if noisy_cls_embed.dim() == 2:
            noisy_cls_embed = noisy_cls_embed.unsqueeze(1)
        
        inputs_embeds = self.model.embed_tokens(input_ids)

        cat_input = torch.cat([noisy_cls_embed, inputs_embeds, noise], dim=1)
        cat_attention_mask = torch.cat([
            torch.ones(noisy_cls_embed.shape[0], 1, device=noisy_cls_embed.device, dtype=torch.long), 
            attention_mask, 
            attention_mask], dim=1)

        transformer_outputs = self.model(
            None,
            attention_mask=cat_attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=cat_input,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        
        denoised_cls_embed = hidden_states[:, 0, :]

        loss = None
        if clean_cls_embed is not None:
            loss_fct = MSELoss()
            loss = loss_fct(denoised_cls_embed, clean_cls_embed)

        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=denoised_cls_embed,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )



class DenoiseModelLlama(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)

        
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        noisy_cls_embed: torch.FloatTensor,
        noise: torch.FloatTensor,
        clean_cls_embed: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        
        return_dict = self.config.use_return_dict
        
        if noisy_cls_embed.dim() == 2:
            noisy_cls_embed = noisy_cls_embed.unsqueeze(1)
        
        inputs_embeds = self.model.embed_tokens(input_ids)

        cat_input = torch.cat([noisy_cls_embed, inputs_embeds, noise], dim=1)
        cat_attention_mask = torch.cat([
            torch.ones(noisy_cls_embed.shape[0], 1, device=noisy_cls_embed.device, dtype=torch.long), 
            attention_mask, 
            attention_mask], dim=1)

        transformer_outputs = self.model(
            None,
            attention_mask=cat_attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=cat_input,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        
        denoised_cls_embed = hidden_states[:, 0, :]

        loss = None
        if clean_cls_embed is not None:
            loss_fct = MSELoss()
            loss = loss_fct(denoised_cls_embed, clean_cls_embed)

        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=denoised_cls_embed,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )




class Qwen2ForSequenceClassification(Qwen2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = Qwen2Model(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        
        denoise_model: Optional[DenoiseModel] = None,
        privacy_budget: Optional[float] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids.cuda())
            inputs_embeds, noise = get_noisy_embedding(
                inputs_embeds,
                privacy_budget,
                _clip_embedding_l2_enabled(),
                model_type="qwen2-1.5b",
            )
        else:
            noise = torch.zeros(inputs_embeds.shape, device=inputs_embeds.device)

        transformer_outputs = self.model(
            None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(hidden_states.device)
            else:
                sequence_lengths = -1

        cls_embeds = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]
        
        if denoise_model is not None:
            denoise_output = denoise_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                noisy_cls_embed=cls_embeds,
                noise=noise,
            )
            cls_embeds = denoise_output.logits
        
        logits = self.score(cls_embeds)
        pooled_logits = logits

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
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
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
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
        output.cls_embeds = cls_embeds
        output.noise = noise
        output.noisy_embeds = inputs_embeds
        return output






class LlamaForSequenceClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        
        denoise_model: Optional[DenoiseModel] = None,
        privacy_budget: Optional[float] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids.cuda())
            inputs_embeds, noise = get_noisy_embedding(
                inputs_embeds,
                privacy_budget,
                _clip_embedding_l2_enabled(),
                model_type="llama-3.2-1b",
            )
        else:
            noise = torch.zeros(inputs_embeds.shape, device=inputs_embeds.device)

        transformer_outputs = self.model(
            None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(hidden_states.device)
            else:
                sequence_lengths = -1

        cls_embeds = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]
        
        if denoise_model is not None:
            denoise_output = denoise_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                noisy_cls_embed=cls_embeds,
                noise=noise,
            )
            cls_embeds = denoise_output.logits
        
        logits = self.score(cls_embeds)
        pooled_logits = logits

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
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
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
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
        output.cls_embeds = cls_embeds
        output.noise = noise
        output.noisy_embeds = inputs_embeds
        return output


