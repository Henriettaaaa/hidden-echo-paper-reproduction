import argparse
import importlib
import json
from pathlib import Path

import datasets
import torch
from peft import PeftModel
from transformers import AutoTokenizer, DataCollatorForSeq2Seq

from modeling.my_t5.configuration import MyT5Config
from modeling.my_t5.split import SplittedT5ForConditionalGeneration


def load_first_available_dataset(candidates):
    errors = []
    for candidate in candidates:
        try:
            if isinstance(candidate, tuple):
                name, config = candidate
                return datasets.load_dataset(name, config)
            return datasets.load_dataset(candidate)
        except Exception as exc:
            errors.append(
                f"{candidate}: {type(exc).__name__}: {str(exc).splitlines()[0][:160]}"
            )
    raise RuntimeError("Unable to load dataset from candidates:\n" + "\n".join(errors))


def load_generation_dataset(dataset_name: str):
    if dataset_name == "dailymail":
        local_path = Path("dataset/cnn_dailymail_short")
        ds = (
            datasets.load_dataset(local_path.as_posix())
            if local_path.exists()
            else load_first_available_dataset(
                ["cnn_dailymail_short", "determined-ai/cnn_dailymail_short"]
            )
        )
        return {
            "train": ds["train"],
            "validation": ds["validation"],
            "test": ds["test"] if "test" in ds else ds["validation"],
            "input_field": "article",
            "label_field": "highlights",
            "input_len": 256,
            "label_len": 96,
            "prefix": "summarize: ",
        }
    raise ValueError(f"Unknown generation dataset: {dataset_name}")


def find_split_model(model):
    stack = [model]
    seen = set()
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, SplittedT5ForConditionalGeneration):
            return current
        stack.append(getattr(current, "base_model", None))
        stack.append(getattr(current, "model", None))
    return None


def load_t5_checkpoint(checkpoint_path: str, model_path: str):
    config = MyT5Config.from_pretrained(Path(checkpoint_path) / "config.json")
    module = importlib.import_module(config.model_cls_module or "modeling.my_t5.split")
    model_cls = getattr(
        module,
        config.model_cls_name or "SplittedT5ForConditionalGeneration",
    )
    model = model_cls.from_pretrained(
        model_path,
        config=config,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
    )
    return PeftModel.from_pretrained(
        model,
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--dataset_name", default="dailymail", choices=["dailymail"])
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_source_length", type=int, default=None)
    parser.add_argument("--max_target_length", type=int, default=None)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    data = load_generation_dataset(args.dataset_name)
    ds = data[args.split]
    end = min(args.sample_index + args.batch_size, len(ds))
    if args.sample_index < 0 or args.sample_index >= len(ds):
        raise IndexError(f"sample_index {args.sample_index} out of range for split size {len(ds)}")
    batch_ds = ds.select(range(args.sample_index, end))

    input_len = args.max_source_length or data["input_len"]
    label_len = args.max_target_length or data["label_len"]
    inputs = [data["prefix"] + text for text in batch_ds[data["input_field"]]]
    model_inputs = tokenizer(
        inputs,
        padding=False,
        max_length=input_len,
        truncation=True,
    )
    labels = tokenizer(
        text_target=batch_ds[data["label_field"]],
        padding=False,
        max_length=label_len,
        truncation=True,
    )["input_ids"]
    labels = [
        [(token if token != tokenizer.pad_token_id else -100) for token in label]
        for label in labels
    ]
    model_inputs["labels"] = labels

    model = load_t5_checkpoint(args.checkpoint_path, args.model_path)
    model = model.to(device).eval()
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding="max_length",
        max_length=input_len,
        label_pad_token_id=-100,
    )
    features = [
        {key: value[idx] for key, value in model_inputs.items()}
        for idx in range(len(batch_ds))
    ]
    batch = collator(features)
    batch = {key: value.to(device) for key, value in batch.items()}

    split_model = find_split_model(model)
    if split_model is None:
        raise RuntimeError("Unable to locate split T5 model for transfer counters")
    split_model.total_embedding_data_transferred = 0
    split_model.total_hidden_states_data_transferred = 0

    with torch.no_grad():
        _ = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

    emb = int(split_model.total_embedding_data_transferred)
    hid = int(split_model.total_hidden_states_data_transferred)
    config = split_model.config
    transmitted_layers = sum(
        1 for idx in range(config.num_layers) if idx not in set(config.lst_skip or [])
    )
    result = {
        "model_path": args.model_path,
        "checkpoint_path": args.checkpoint_path,
        "dataset_name": args.dataset_name,
        "split": args.split,
        "sample_index": args.sample_index,
        "batch_size": int(batch["input_ids"].shape[0]),
        "sequence_length": int(batch["input_ids"].shape[1]),
        "target_sequence_length": int(batch["labels"].shape[1]),
        "num_layers": int(config.num_layers),
        "transmitted_layers": int(transmitted_layers),
        "lst_reduce_factor": int(config.lst_reduce_factor),
        "reduced_hidden_size": int(config.d_model // config.lst_reduce_factor),
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
