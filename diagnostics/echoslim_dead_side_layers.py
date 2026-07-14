import argparse
import csv
import json
import sys
from pathlib import Path

import datasets
import torch
import transformers
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# 诊断脚本位于 diagnostics/ 子目录。直接以文件路径运行时，Python 只会把
# diagnostics/ 加入 sys.path，而不会自动加入项目根目录；所有本项目脚本都要
# 先显式加入 repo root，再 import modeling/utils。
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modeling.my.configuration import AdditionalConfig
from modeling.my.split import SplittedQwen2ForSequenceClassification
from utils.model import get_base_classification_model_for_training, setup_seed


def get_financial_phrasebank_splits(config_name: str):
    ds = datasets.load_dataset("financial_phrasebank", config_name, revision="main")["train"]
    if config_name != "sentences_allagree":
        raise ValueError("This diagnostic currently expects sentences_allagree")
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


def module_param_count(module):
    if module is None:
        return 0
    return sum(p.numel() for p in module.parameters())


def grad_summary_for_params(params):
    total_params = 0
    grad_params = 0
    grad_numel = 0
    grad_sq_sum = 0.0
    max_abs_grad = 0.0
    for param in params:
        total_params += param.numel()
        if param.grad is None:
            continue
        grad_params += param.numel()
        grad_numel += param.grad.numel()
        grad = param.grad.detach().float()
        grad_sq_sum += float(torch.sum(grad * grad).item())
        max_abs_grad = max(max_abs_grad, float(grad.abs().max().item()))
    return {
        "total_params": total_params,
        "grad_params": grad_params,
        "grad_numel": grad_numel,
        "grad_norm": grad_sq_sum ** 0.5,
        "max_abs_grad": max_abs_grad,
        "has_grad": grad_numel > 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--privacy_budget", type=float, default=5000.0)
    parser.add_argument("--num_reserved_layers", type=int, default=3)
    parser.add_argument("--keep_last_layer", type=lambda x: x.lower() in ["true", "1", "yes"], default=True)
    parser.add_argument("--lst_reduce_factor", type=int, default=4)
    parser.add_argument("--num_integrate_step", type=int, default=5)
    parser.add_argument("--num_samples", type=int, default=32)
    parser.add_argument("--num_integrate_batch_size", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_probe_batches", type=int, default=2)
    parser.add_argument("--max_len", type=int, default=96)
    args = parser.parse_args()

    setup_seed(12399)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = AdditionalConfig(
        privacy_budget=args.privacy_budget,
        clip_embedding_l2=True,
        noise_type="Chi",
        lst_enable=True,
        lst_reduce_factor=args.lst_reduce_factor,
        lst_skip=[-1],
        lst_input_type="clean",
        lst_random_init=False,
        auto_skip=True,
        num_reserved_layers=args.num_reserved_layers,
        num_integrate_step=args.num_integrate_step,
        num_samples=args.num_samples,
        keep_last_layer=args.keep_last_layer,
        num_integrate_batch_size=args.num_integrate_batch_size,
        mi_downsample_enable=True,
        mi_estimator_iter_num=2,
        mi_estimator_lr=1e-4,
        mi_xz_ratio=0.001,
        mi_yz_ratio=0.001,
        mi_estimator_hidden_dim=128,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    ds = get_financial_phrasebank_splits("sentences_allagree")

    def preprocess_function(examples):
        return tokenizer(
            examples["sentence"],
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            max_length=args.max_len,
        )

    model = get_base_classification_model_for_training(
        num_labels=3,
        custom_config=config,
        model_cls=SplittedQwen2ForSequenceClassification,
        model_path=args.model_path,
    )

    sample_datasets = ds["train"].train_test_split(
        test_size=config.num_samples,
        stratify_by_column="label",
        seed=123,
    )["test"]
    tokenized_sample_dataset = sample_datasets.map(preprocess_function, batched=True)
    sample_dataloader = DataLoader(
        tokenized_sample_dataset,
        batch_size=config.num_integrate_batch_size,
        collate_fn=transformers.default_data_collator,
    )
    layer_idx_sorted_by_attribution = model.calc_layer_attributions(sample_dataloader)
    model.set_layer_skip(
        layer_idx_sorted_by_attribution,
        config.num_reserved_layers,
        config.keep_last_layer,
    )

    # 注意：训练入口会经由 Trainer/PEFT 统一处理设备，但这个诊断脚本直接裸调用
    # split model。from_pretrained(device_map="cuda") 只保证 backbone 在 CUDA，
    # 新增的 downsample/denoise/head 可能仍在 CPU；forward 前必须显式对齐。
    model_device = next(model.server_backbone.parameters()).device
    model_dtype = next(model.server_backbone.parameters()).dtype
    model.server_downsample.to(device=model_device, dtype=model_dtype)
    model.client_denoise.to(device=model_device, dtype=model_dtype)
    model.client_head.to(device=model_device, dtype=model_dtype)

    selected_layers = [
        idx
        for idx in range(model.config.num_hidden_layers)
        if idx not in set(model.config.lst_skip or [])
    ]
    skipped_layers = [
        idx
        for idx in range(model.config.num_hidden_layers)
        if idx in set(model.config.lst_skip or [])
    ]

    # 诊断只关心 HE+ client side stack。冻结其它参数可以降低显存，并让梯度统计更清楚。
    for param in model.parameters():
        param.requires_grad_(False)
    for param in model.client_denoise.parameters():
        param.requires_grad_(True)
    for param in model.client_head.parameters():
        param.requires_grad_(True)

    tokenized_train_dataset = ds["train"].map(preprocess_function, batched=True)
    probe_loader = DataLoader(
        tokenized_train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=transformers.default_data_collator,
    )

    model.train()
    model.zero_grad(set_to_none=True)
    losses = []
    for batch_idx, batch in enumerate(probe_loader):
        if batch_idx >= args.num_probe_batches:
            break
        batch = {
            key: value.to(model.device) if hasattr(value, "to") else value
            for key, value in batch.items()
        }
        outputs = model(**batch)
        loss = outputs.loss
        losses.append(float(loss.detach().float().item()))
        loss.backward()

    trans = model.client_denoise.ladder_side.trans
    rows = []
    selected_set = set(selected_layers)
    skipped_set = set(skipped_layers)
    for layer_idx, layer in enumerate(trans.dec_layers):
        params = list(layer.parameters())
        gate_param = trans.gate_vectors.gate_vectors[layer_idx]
        params_with_gate = params + [gate_param]
        row = {
            "layer_idx": layer_idx,
            "selected": layer_idx in selected_set,
            "skipped": layer_idx in skipped_set,
            "layer_param_count": module_param_count(layer),
            "gate_param_count": gate_param.numel(),
        }
        row.update(grad_summary_for_params(params_with_gate))
        rows.append(row)

    selected_grad_layers = [r for r in rows if r["selected"] and r["has_grad"]]
    skipped_grad_layers = [r for r in rows if r["skipped"] and r["has_grad"]]
    selected_param_count = sum(r["total_params"] for r in rows if r["selected"])
    skipped_param_count = sum(r["total_params"] for r in rows if r["skipped"])
    total_param_count = sum(r["total_params"] for r in rows)

    summary = {
        "diagnostic": "hiddenecho_plus_dead_side_layers",
        "privacy_budget": args.privacy_budget,
        "num_reserved_layers": args.num_reserved_layers,
        "keep_last_layer": args.keep_last_layer,
        "selected_layers": selected_layers,
        "skipped_layers": skipped_layers,
        "num_probe_batches": args.num_probe_batches,
        "probe_batch_size": args.batch_size,
        "losses": losses,
        "total_side_layer_and_gate_params": total_param_count,
        "selected_side_layer_and_gate_params": selected_param_count,
        "skipped_side_layer_and_gate_params": skipped_param_count,
        "skipped_param_fraction": skipped_param_count / total_param_count,
        "selected_layers_with_grad": [r["layer_idx"] for r in selected_grad_layers],
        "skipped_layers_with_grad": [r["layer_idx"] for r in skipped_grad_layers],
        "num_skipped_layers_with_grad": len(skipped_grad_layers),
        "all_skipped_layers_are_gradient_dead": len(skipped_grad_layers) == 0,
    }

    with (output_dir / "dead_side_layers_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    with (output_dir / "dead_side_layers_layer_stats.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md = [
        "# HE+ Dead Side Layers Diagnostic",
        "",
        f"- selected_layers: `{selected_layers}`",
        f"- skipped_layers: `{skipped_layers}`",
        f"- skipped_param_fraction: `{summary['skipped_param_fraction']:.4f}`",
        f"- selected_layers_with_grad: `{summary['selected_layers_with_grad']}`",
        f"- skipped_layers_with_grad: `{summary['skipped_layers_with_grad']}`",
        f"- all_skipped_layers_are_gradient_dead: `{summary['all_skipped_layers_are_gradient_dead']}`",
        "",
        "| group | layers | params | grad layers |",
        "|---|---:|---:|---|",
        f"| selected | {len(selected_layers)} | {selected_param_count} | `{summary['selected_layers_with_grad']}` |",
        f"| skipped | {len(skipped_layers)} | {skipped_param_count} | `{summary['skipped_layers_with_grad']}` |",
    ]
    (output_dir / "dead_side_layers_summary.md").write_text("\n".join(md) + "\n")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
