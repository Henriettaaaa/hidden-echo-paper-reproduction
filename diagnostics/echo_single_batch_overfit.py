import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import sklearn.metrics
import torch
import datasets
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from transformers import AutoTokenizer

from modeling.my.configuration import AdditionalConfig
from modeling.my.split import SplittedQwen2ForSequenceClassification
from utils.model import get_base_classification_model_for_training, setup_seed


def get_financial_phrasebank_splits(config_name: str):
    ds = datasets.load_dataset("financial_phrasebank", config_name, revision="main")["train"]

    if config_name == "sentences_50agree":
        split = ds.train_test_split(test_size=0.1, seed=123, stratify_by_column="label")
        dev_test = split["test"].train_test_split(test_size=0.5, seed=123)
        return {
            "train": split["train"],
            "validation": dev_test["train"],
            "test": dev_test["test"],
        }

    if config_name == "sentences_allagree":
        expected_size = 1811 + 226 + 227
        if len(ds) != expected_size:
            raise ValueError(
                f"Expected financial_phrasebank/{config_name} to contain "
                f"{expected_size} rows, got {len(ds)}"
            )
        split = ds.train_test_split(
            train_size=1811,
            test_size=226 + 227,
            seed=123,
            stratify_by_column="label",
        )
        dev_test = split["test"].train_test_split(
            train_size=226,
            test_size=227,
            seed=123,
            stratify_by_column="label",
        )
        return {
            "train": split["train"],
            "validation": dev_test["train"],
            "test": dev_test["test"],
        }

    raise ValueError(f"Unknown financial_phrasebank config: {config_name}")


def build_balanced_batch(dataset, per_class):
    selected = []
    counts = {}
    for idx, label in enumerate(dataset["label"]):
        counts.setdefault(label, 0)
        if counts[label] < per_class:
            selected.append(idx)
            counts[label] += 1
        if len(counts) == 3 and all(v >= per_class for v in counts.values()):
            break
    return dataset.select(selected)


def disable_dropout(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0


def reset_noise_seed(seed):
    if seed is None:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics(logits, labels):
    probs = torch.softmax(logits.float(), dim=-1).detach().cpu().numpy()
    preds = probs.argmax(axis=-1)
    y = labels.detach().cpu().numpy()
    metrics = {
        "accuracy": float(sklearn.metrics.accuracy_score(y, preds)),
        "macro_f1": float(sklearn.metrics.f1_score(y, preds, average="macro")),
    }
    try:
        metrics["auc"] = float(sklearn.metrics.roc_auc_score(y, probs, multi_class="ovr"))
    except ValueError:
        metrics["auc"] = None
    return metrics


def get_split_model(model):
    base_model = getattr(model, "base_model", model)
    return getattr(base_model, "model", base_model)


def masked_mse(a, b, attention_mask):
    a = a.float()
    b = b.float()
    mask = attention_mask.unsqueeze(-1).to(device=a.device, dtype=a.dtype)
    denom = mask.sum().clamp_min(1.0) * a.shape[-1]
    return (((a - b) ** 2) * mask).sum() / denom


def compute_hidden_mse(model, batch, fixed_noise_seed):
    split_model = get_split_model(model)
    if not split_model.config.lst_enable:
        return {}

    reset_noise_seed(fixed_noise_seed)
    clean_input_embeds, noisy_input_embeds = split_model.client_embedding(batch["input_ids"])
    attention_mask = batch["attention_mask"]

    clean_outputs = split_model.server_backbone(
        None,
        attention_mask=attention_mask,
        inputs_embeds=clean_input_embeds,
        output_hidden_states=True,
        return_dict=True,
    )
    noisy_outputs = split_model.server_backbone(
        None,
        attention_mask=attention_mask,
        inputs_embeds=noisy_input_embeds,
        output_hidden_states=True,
        return_dict=True,
    )

    all_hidden_states = noisy_outputs.hidden_states[1:]
    all_hidden_states = split_model.server_layer_select(all_hidden_states)
    all_hidden_states = split_model.server_downsample(all_hidden_states)
    denoised_hidden, _ = split_model.client_denoise(
        all_hidden_states,
        attention_mask,
        clean_input_embeds,
        noisy_input_embeds,
        output_hidden_states=False,
    )

    clean_hidden = clean_outputs.last_hidden_state
    noisy_hidden = noisy_outputs.last_hidden_state
    noisy_mse = masked_mse(noisy_hidden, clean_hidden, attention_mask)
    denoised_mse = masked_mse(denoised_hidden, clean_hidden, attention_mask)
    return {
        "mse_noisy_to_clean": float(noisy_mse.detach().cpu()),
        "mse_denoised_to_clean": float(denoised_mse.detach().cpu()),
        "mse_denoised_over_noisy": float((denoised_mse / noisy_mse.clamp_min(1e-12)).detach().cpu()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/data1/models/models--Qwen--Qwen2.5-1.5B-Instruct")
    parser.add_argument("--financial_phrasebank_config", default="sentences_allagree")
    parser.add_argument("--privacy_budget", type=float, default=1000.0)
    parser.add_argument("--clip_embedding_l2", type=lambda x: x.lower() in ["true", "1", "yes"], default=True)
    parser.add_argument("--lst_enable", type=lambda x: x.lower() in ["true", "1", "yes"], default=True)
    parser.add_argument("--lst_reduce_factor", type=int, default=16)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--per_class", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--fixed_noise_seed", type=int, default=12345)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    setup_seed(12399)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    ds = get_financial_phrasebank_splits(args.financial_phrasebank_config)["train"]
    batch_ds = build_balanced_batch(ds, args.per_class)
    tokenized = tokenizer(
        batch_ds["sentence"],
        truncation=True,
        padding="max_length",
        return_tensors="pt",
        max_length=96,
    )
    labels = torch.tensor(batch_ds["label"], dtype=torch.long)
    tokenized["labels"] = labels

    custom_config = AdditionalConfig(
        privacy_budget=args.privacy_budget,
        clip_embedding_l2=args.clip_embedding_l2,
        noise_type="Chi",
        lst_enable=args.lst_enable,
        lst_reduce_factor=args.lst_reduce_factor,
        lst_input_type="clean",
        lst_skip=[-1],
        lst_random_init=False,
        auto_skip=False,
        mi_downsample_enable=False,
        use_residual=True,
    )

    model = get_base_classification_model_for_training(
        num_labels=3,
        tokenizer=tokenizer,
        custom_config=custom_config,
        model_path=args.model_path,
        model_cls=SplittedQwen2ForSequenceClassification,
    )

    modules_to_save = ["client_denoise", "server_downsample"] if args.lst_enable else []
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=args.lora_rank,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "gate_proj",
            "down_proj",
        ],
        bias="none",
        modules_to_save=modules_to_save,
        layers_to_transform=list(range(model.config.num_hidden_layers)),
        layers_pattern="layers",
    )
    model = get_peft_model(model, lora_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    disable_dropout(model)
    model.train()

    batch = {k: v.to(device) for k, v in tokenized.items()}
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate, weight_decay=0.0)

    history = []
    for step in range(1, args.steps + 1):
        reset_noise_seed(args.fixed_noise_seed)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                reset_noise_seed(args.fixed_noise_seed)
                eval_outputs = model(**batch)
                metrics = compute_metrics(eval_outputs.logits, batch["labels"])
                hidden_mse = compute_hidden_mse(model, batch, args.fixed_noise_seed)
            model.train()
            row = {
                "step": step,
                "loss": float(eval_outputs.loss.detach().float().cpu()),
                **metrics,
                **hidden_mse,
            }
            history.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "args": vars(args),
                    "labels": batch_ds["label"],
                    "history": history,
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
