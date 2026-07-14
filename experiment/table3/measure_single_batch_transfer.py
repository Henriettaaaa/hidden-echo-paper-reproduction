import argparse
import json
from pathlib import Path

import datasets
import torch
from transformers import AutoTokenizer

from modeling.my.configuration import MyQwen2Config
from modeling.my.split import SplittedQwen2ForSequenceClassification
from utils.model import load_classification_model_checkpoint


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
                f"Expected financial_phrasebank/{config_name} to contain {expected_size} rows, got {len(ds)}"
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


def find_split_model(model):
    stack = [model]
    seen = set()
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, SplittedQwen2ForSequenceClassification):
            return current
        stack.append(getattr(current, "base_model", None))
        stack.append(getattr(current, "model", None))
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument(
        "--financial_phrasebank_config",
        default="sentences_allagree",
        choices=["sentences_50agree", "sentences_allagree"],
    )
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_len", type=int, default=96)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    ds = get_financial_phrasebank_splits(args.financial_phrasebank_config)[args.split]
    end = min(args.sample_index + args.batch_size, len(ds))
    if args.sample_index < 0 or args.sample_index >= len(ds):
        raise IndexError(f"sample_index {args.sample_index} out of range for split size {len(ds)}")
    batch_ds = ds.select(range(args.sample_index, end))

    batch = tokenizer(
        batch_ds["sentence"],
        truncation=True,
        padding="max_length",
        return_tensors="pt",
        max_length=args.max_len,
    )
    batch = {k: v.to(device) for k, v in batch.items()}

    model = load_classification_model_checkpoint(
        args.checkpoint_path,
        tokenizer=tokenizer,
        num_labels=3,
        model_cls=SplittedQwen2ForSequenceClassification,
        model_path=args.model_path,
        config_cls=MyQwen2Config,
    )
    # PEFT may restore modules_to_save on CPU even when the backbone uses
    # device_map="cuda"; move the complete wrapper before the measured forward.
    model = model.to(device).eval()

    split_model = find_split_model(model)
    if split_model is None:
        raise RuntimeError("Unable to locate split model for transfer counters")
    split_model.total_embedding_data_transferred = 0
    split_model.total_hidden_states_data_transferred = 0

    with torch.no_grad():
        _ = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch.get("labels"),
        )

    emb = int(split_model.total_embedding_data_transferred)
    hid = int(split_model.total_hidden_states_data_transferred)
    result = {
        "model_path": args.model_path,
        "checkpoint_path": args.checkpoint_path,
        "financial_phrasebank_config": args.financial_phrasebank_config,
        "split": args.split,
        "sample_index": args.sample_index,
        "batch_size": args.batch_size,
        "sequence_length": int(batch["input_ids"].shape[1]),
        "embedding_data_transferred": emb,
        "hiddens_data_transferred": hid,
        "embedding_mib": emb / 1024 / 1024,
        "hiddens_mib": hid / 1024 / 1024,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
