import importlib
import torch
from tqdm import tqdm
from peft import PeftModel
import os
import gc
import numpy as np
import random
from typing import TypedDict
from typing_extensions import NotRequired
import argparse
from dataclasses import dataclass
from transformers import AutoTokenizer, PreTrainedModel

import transformers.models.qwen2.modeling_qwen2
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

from modeling.my.configuration import MyQwen2Config, AdditionalConfig
from modeling.my.split import SplittedQwen2ForSequenceClassification

# 模型加载、checkpoint 恢复、参数解析、随机种子

def get_base_classification_model_for_training(
    num_labels: int = 2,
    *,
    tokenizer = None,
    custom_config: AdditionalConfig | None = None,
    model_path: str = "Qwen2-1.5B-Instruct",
    model_cls: type[PreTrainedModel] = SplittedQwen2ForSequenceClassification,
    config_cls = MyQwen2Config,
):
    config = config_cls.from_pretrained(
        os.path.join(model_path, "config.json"),
        num_labels=num_labels,
        pad_token_id=151643 if tokenizer is None else tokenizer.pad_token_id,
        **custom_config.__dict__ if custom_config is not None else {},
    )

    base_model = model_cls.from_pretrained(
        model_path,
        config=config,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
    )

    base_model.config.model_cls_module = base_model.__class__.__module__
    base_model.config.model_cls_name = base_model.__class__.__name__

    return base_model


def get_raw_causal_model(
    model_path: str = "Qwen2-1.5B-Instruct",
):
    causal_model = Qwen2ForCausalLM.from_pretrained(
        model_path,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
    )
    causal_model.config.pad_token_id = 151643
    return causal_model


def get_tokenizer(
    model_path: str = "Qwen2-1.5B-Instruct",
):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return tokenizer


def load_classification_model_checkpoint(
    checkpoint_path: str,
    *,
    tokenizer = None,
    num_labels: int = 2,
    model_cls: type[PreTrainedModel] = SplittedQwen2ForSequenceClassification,
    model_path: str = "Qwen2-1.5B-Instruct",
    config_cls = MyQwen2Config,
):
    config = config_cls.from_pretrained(
        os.path.join(checkpoint_path, "config.json"),
        num_labels=num_labels,
        pad_token_id=151643 if tokenizer is None else tokenizer.pad_token_id,
    )

    module = transformers.models.qwen2.modeling_qwen2
    if config.model_cls_module is not None:
        module = importlib.import_module(config.model_cls_module)

    if config.model_cls_name is not None:
        model_cls = getattr(module, config.model_cls_name)
        print(f"Using model class: {module}.{model_cls}")

    model = model_cls.from_pretrained(
        model_path,
        config=config,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
    )

    lora_model = PeftModel.from_pretrained(
        model,
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        # load_in_4bit=True,
        device_map="cuda",
    )

    return lora_model


@dataclass
class TrainArgs:
    experiment_name: str
    dataset_name: str
    financial_phrasebank_config: str
    model_path: str
    num_train_epochs: int
    learning_rate: float
    max_len: int
    train_batch_size: int
    eval_batch_size: int
    lora_rank: int
    model_options: AdditionalConfig
    lr_scheduler_type: str = "warmup_stable_decay"
    lr_scheduler_wsd_decay_epochs: int = 3


def parse_args_for_model_train_options():
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment_name", type=str, default="default")
    parser.add_argument("--dataset_name", type=str, default="mrpc")
    parser.add_argument(
        "--financial_phrasebank_config",
        type=str,
        default="sentences_50agree",
        choices=["sentences_50agree", "sentences_allagree"],
    )
    parser.add_argument("--model_path", type=str, default="Qwen2-1.5B-Instruct")

    parser.add_argument("--num_train_epochs", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--lora_rank", type=int, default=16)
    
    parser.add_argument("--lr_scheduler_type", type=str, default="warmup_stable_decay")
    parser.add_argument("--lr_scheduler_wsd_decay_epochs", type=int, default=3)

    for key, value in AdditionalConfig().__dict__.items():
        if isinstance(value, list):
            parser.add_argument(f"--{key}", nargs="+", type=type(value[0]), default=value)
        elif isinstance(value, bool):
            parser.add_argument(f"--{key}", type=lambda x: x.lower() in ["true", "1", "yes"], default=value)
        else:
            parser.add_argument(f"--{key}", type=type(value), default=value)

    args = parser.parse_args()

    train_args = TrainArgs(
        experiment_name=args.experiment_name,
        dataset_name=args.dataset_name,
        financial_phrasebank_config=args.financial_phrasebank_config,
        model_path=args.model_path,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        max_len=args.max_len,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        lora_rank=args.lora_rank,
        model_options=AdditionalConfig(
            **{key: getattr(args, key) for key in AdditionalConfig().__dict__.keys()}
        ),
        
        lr_scheduler_type=args.lr_scheduler_type,
        lr_scheduler_wsd_decay_epochs=args.lr_scheduler_wsd_decay_epochs,
    )

    return train_args


def predict_model(model, batched_tokenized_dataset) -> list:
    predictions = []
    for batch in tqdm(batched_tokenized_dataset):
        with torch.no_grad():
            outputs = model(**batch)
        
        logits = outputs.logits

        predicted_labels = torch.argmax(logits, dim=1).tolist()
        predictions.extend(predicted_labels)

    return predictions


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
