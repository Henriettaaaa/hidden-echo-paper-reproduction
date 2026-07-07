import argparse
import json
import os
from pathlib import Path

os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_DATASETS_CACHE"] = os.environ.get(
    "HF_DATASETS_CACHE", "/data/songhanlin/tmp/hf-datasets"
)
os.environ["HF_HOME"] = os.environ.get("HF_HOME", "/data/songhanlin/tmp/hf-home")
os.environ["TRANSFORMERS_CACHE"] = os.environ.get(
    "TRANSFORMERS_CACHE", "/data/songhanlin/tmp/hf-cache"
)

import datasets
import evaluate
import numpy as np
import torch
import transformers
from matplotlib import pyplot as plt
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from typing_extensions import override

from modeling.my.configuration import AdditionalConfig
from modeling.my_t5.configuration import MyT5Config
from modeling.my_t5.split import SplittedT5ForConditionalGeneration
from utils.model import setup_seed


setup_seed(12399)


def load_first_available_dataset(candidates):
    errors = []
    for candidate in candidates:
        try:
            if isinstance(candidate, tuple):
                name, config = candidate
                return datasets.load_dataset(name, config)
            return datasets.load_dataset(candidate)
        except Exception as exc:
            errors.append(f"{candidate}: {type(exc).__name__}: {str(exc).splitlines()[0][:160]}")
    raise RuntimeError("Unable to load dataset from candidates:\n" + "\n".join(errors))


def load_generation_dataset(dataset_name: str):
    if dataset_name == "dailymail":
        local_path = Path("dataset/cnn_dailymail_short")
        ds = (
            datasets.load_dataset(local_path.as_posix())
            if local_path.exists()
            else load_first_available_dataset(
                [
                    "cnn_dailymail_short",
                    "determined-ai/cnn_dailymail_short",
                ]
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

    if dataset_name == "samsum":
        local_path = Path("dataset/samsum")
        ds = (
            datasets.load_dataset(local_path.as_posix())
            if local_path.exists()
            else load_first_available_dataset(
                [
                    "samsum",
                    "Samsung/samsum",
                    "knkarthick/samsum",
                    "ccdv/samsum",
                ]
            )
        )
        return {
            "train": ds["train"],
            "validation": ds["validation"],
            "test": ds["test"] if "test" in ds else ds["validation"],
            "input_field": "dialogue",
            "label_field": "summary",
            "input_len": 80,
            "label_len": 50,
            "prefix": "summarize: ",
        }

    if dataset_name == "fr2en":
        csv_path = Path("dataset/damo_mt_testsets_fr2en_iwslt14.csv")
        if not csv_path.exists():
            raise FileNotFoundError(
                "Expected IWSLT csv at dataset/damo_mt_testsets_fr2en_iwslt14.csv"
            )
        ds = datasets.Dataset.from_csv(
            csv_path.as_posix(),
            column_names=["en", "fr"],
        )
        split = ds.train_test_split(test_size=0.1, seed=123)
        return {
            "train": split["train"],
            "validation": split["test"],
            "test": split["test"],
            "input_field": "en",
            "label_field": "fr",
            "input_len": 128,
            "label_len": 128,
            "prefix": "translate English to French: ",
        }

    raise ValueError(f"Unknown generation dataset: {dataset_name}")


class MyTrainer(Seq2SeqTrainer):
    @override
    def save_model(self, output_dir=None, _internal_call=False):
        self.model.config.save_pretrained(output_dir)

        train_logs = [log for log in self.state.log_history if "loss" in log]
        if train_logs:
            x = [log["epoch"] for log in train_logs]
            y = [log["loss"] for log in train_logs]
            plt.plot(x, y, label="train_loss")

            eval_logs = [log for log in self.state.log_history if "eval_loss" in log]
            if eval_logs:
                x = [log["epoch"] for log in eval_logs]
                y = [log["eval_loss"] for log in eval_logs]
                plt.plot(x, y, label="eval_loss")

            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Loss Curve")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{output_dir}/loss_curve.png")
            plt.close()

        return super().save_model(output_dir, _internal_call)


def get_base_t5_model_for_training(
    *,
    custom_config: MyT5Config,
    model_path: str,
):
    additional_config = {
        key: getattr(custom_config, key)
        for key in AdditionalConfig().__dict__.keys()
        if hasattr(custom_config, key)
    }
    config = MyT5Config.from_pretrained(
        os.path.join(model_path, "config.json")
        if Path(model_path, "config.json").exists()
        else model_path,
        **additional_config,
    )
    model = SplittedT5ForConditionalGeneration.from_pretrained(
        model_path,
        config=config,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
    )
    model.config.model_cls_module = model.__class__.__module__
    model.config.model_cls_name = model.__class__.__name__
    return model


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


def train(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    data = load_generation_dataset(args.dataset_name)

    train_dataset = data["train"]
    eval_dataset = data["validation"]
    test_dataset = data["test"]
    if args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(min(args.max_train_samples, len(train_dataset))))
    if args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(min(args.max_eval_samples, len(eval_dataset))))
        test_dataset = test_dataset.select(range(min(args.max_eval_samples, len(test_dataset))))

    input_field = data["input_field"]
    label_field = data["label_field"]
    input_len = args.max_source_length or data["input_len"]
    label_len = args.max_target_length or data["label_len"]
    prefix = data["prefix"]

    def preprocess_function(examples):
        inputs = [prefix + text for text in examples[input_field]]
        model_inputs = tokenizer(
            inputs,
            padding=False,
            max_length=input_len,
            truncation=True,
        )
        labels = tokenizer(
            text_target=examples[label_field],
            padding=False,
            max_length=label_len,
            truncation=True,
        )["input_ids"]
        labels = [
            [(token if token != tokenizer.pad_token_id else -100) for token in label]
            for label in labels
        ]
        model_inputs["labels"] = labels
        return model_inputs

    tokenized_train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    tokenized_eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=eval_dataset.column_names,
    )
    tokenized_test_dataset = test_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=test_dataset.column_names,
    )

    custom_config = MyT5Config(
        privacy_budget=args.privacy_budget,
        clip_embedding_l2=args.clip_embedding_l2,
        noise_type=args.noise_type,
        lst_reduce_factor=args.lst_reduce_factor,
        lst_skip=args.lst_skip,
        lst_temperature=args.lst_temperature,
        lst_input_type=args.lst_input_type,
        lst_enable=args.lst_enable,
        lst_random_init=args.lst_random_init,
        auto_skip=args.auto_skip,
        num_reserved_layers=args.num_reserved_layers,
        num_integrate_step=args.num_integrate_step,
        num_samples=args.num_samples,
        keep_last_layer=args.keep_last_layer,
        num_integrate_batch_size=args.num_integrate_batch_size,
        mi_downsample_enable=args.mi_downsample_enable,
        mi_estimator_iter_num=args.mi_estimator_iter_num,
        mi_estimator_lr=args.mi_estimator_lr,
        mi_xz_ratio=args.mi_xz_ratio,
        mi_yz_ratio=args.mi_yz_ratio,
        mi_estimator_hidden_dim=args.mi_estimator_hidden_dim,
        use_residual=args.use_residual,
    )
    custom_config.lst_skip = list(custom_config.lst_skip or [])

    model = get_base_t5_model_for_training(
        custom_config=custom_config,
        model_path=args.model_path,
    )

    if custom_config.auto_skip:
        sample_size = min(custom_config.num_samples, len(tokenized_train_dataset))
        sample_dataset = tokenized_train_dataset.select(range(sample_size))
        sample_dataloader = DataLoader(
            sample_dataset,
            batch_size=custom_config.num_integrate_batch_size,
            collate_fn=DataCollatorForSeq2Seq(
                tokenizer=tokenizer,
                model=model,
                padding=True,
                label_pad_token_id=-100,
            ),
        )
        layer_idx_sorted_by_attribution = model.calc_layer_attributions(
            sample_dataloader
        )
        print(f"layer_idx_sorted_by_attribution: {layer_idx_sorted_by_attribution}")
        model.set_layer_skip(
            layer_idx_sorted_by_attribution,
            custom_config.num_reserved_layers,
            custom_config.keep_last_layer,
        )

    modules_to_save = []
    if custom_config.lst_enable:
        modules_to_save.extend(["client_denoise", "server_downsample"])

    if args.lora_scope == "encoder_decoder":
        target_modules = r"^(server_encoder|decoder)\..*\.(q|v)$"
    elif args.lora_scope == "encoder":
        target_modules = r"^server_encoder\..*\.(q|v)$"
    elif args.lora_scope == "decoder":
        target_modules = r"^decoder\..*\.(q|v)$"
    else:
        raise ValueError(f"Unknown lora_scope: {args.lora_scope}")

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=args.lora_rank,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none",
        modules_to_save=modules_to_save or None,
    )
    model = get_peft_model(model, lora_config)
    print(model)

    steps_per_epoch = max(1, len(tokenized_train_dataset) // args.train_batch_size)
    total_steps = steps_per_epoch * args.num_train_epochs
    lr_scheduler_kwargs = {}
    if args.lr_scheduler_type == "warmup_stable_decay":
        decay_steps = args.lr_scheduler_wsd_decay_epochs * steps_per_epoch
        lr_scheduler_kwargs = {
            "num_stable_steps": max(0, total_steps - decay_steps),
            "num_decay_steps": decay_steps,
        }

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        logging_dir=args.output_dir,
        logging_steps=args.logging_steps,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=2,
        use_cpu=False,
        seed=123,
        lr_scheduler_type=args.lr_scheduler_type,
        lr_scheduler_kwargs=lr_scheduler_kwargs,
        remove_unused_columns=False,
        predict_with_generate=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=True,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )
    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    split_model = find_split_model(trainer.model)
    if split_model is not None:
        print(f"embedding_data_transferred: {split_model.total_embedding_data_transferred}")
        print(f"hiddens_data_transferred: {split_model.total_hidden_states_data_transferred}")
        with open(Path(args.output_dir) / "data_transfer.txt", "w") as f:
            f.write(
                f"embedding_data_transferred: {split_model.total_embedding_data_transferred}\n"
            )
            f.write(
                f"hiddens_data_transferred: {split_model.total_hidden_states_data_transferred}\n"
            )

    eval_results = {}
    eval_loader = DataLoader(
        tokenized_test_dataset,
        batch_size=args.eval_batch_size,
        collate_fn=data_collator,
    )

    trained_model = trainer.model.eval()
    for _ in range(args.eval_repeats):
        bleu_metric = evaluate.load("sacrebleu")
        rouge_metric = evaluate.load("rouge")
        for batch in tqdm(eval_loader):
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            generated_tokens = trained_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.generation_max_new_tokens,
                num_beams=args.num_beams,
            )
            predictions = tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True,
            )

            label_ids = batch["labels"].cpu().numpy()
            label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)
            references = tokenizer.batch_decode(
                label_ids,
                skip_special_tokens=True,
            )
            bleu_metric.add_batch(
                predictions=predictions,
                references=[[ref] for ref in references],
            )
            rouge_metric.add_batch(predictions=predictions, references=references)

        eval_results.setdefault("bleu", []).append(bleu_metric.compute()["score"])
        for key, value in rouge_metric.compute().items():
            eval_results.setdefault(key, []).append(value)

    avg_results = {key: float(np.mean(value)) for key, value in eval_results.items()}
    print(avg_results)
    with open(Path(args.output_dir) / "eval_results.json", "w") as f:
        json.dump(avg_results, f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="t5_generation")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="samsum",
        choices=["fr2en", "dailymail", "samsum"],
    )
    parser.add_argument("--model_path", type=str, default="t5-large")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_train_epochs", type=int, default=15)
    parser.add_argument("--learning_rate", type=float, default=1.5e-4)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument(
        "--lora_scope",
        type=str,
        default="encoder_decoder",
        choices=["encoder_decoder", "encoder", "decoder"],
    )
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_strategy", type=str, default="epoch")
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--evaluation_strategy", type=str, default="epoch")
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--lr_scheduler_wsd_decay_epochs", type=int, default=3)
    parser.add_argument("--max_source_length", type=int, default=None)
    parser.add_argument("--max_target_length", type=int, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--eval_repeats", type=int, default=5)
    parser.add_argument("--generation_max_new_tokens", type=int, default=64)
    parser.add_argument("--num_beams", type=int, default=1)

    parser.add_argument("--privacy_budget", type=float, default=30.0)
    parser.add_argument("--clip_embedding_l2", type=lambda x: x.lower() in ["true", "1", "yes"], default=True)
    parser.add_argument("--noise_type", type=str, default="Chi", choices=["Chi", "Gaussian"])
    parser.add_argument("--lst_reduce_factor", type=int, default=8)
    parser.add_argument("--lst_skip", nargs="+", type=int, default=[-1])
    parser.add_argument("--lst_temperature", type=float, default=0.1)
    parser.add_argument("--lst_input_type", type=str, default="clean", choices=["clean", "noisy"])
    parser.add_argument("--lst_enable", type=lambda x: x.lower() in ["true", "1", "yes"], default=True)
    parser.add_argument("--lst_random_init", type=lambda x: x.lower() in ["true", "1", "yes"], default=False)
    parser.add_argument("--auto_skip", type=lambda x: x.lower() in ["true", "1", "yes"], default=False)
    parser.add_argument("--num_reserved_layers", type=int, default=3)
    parser.add_argument("--num_integrate_step", type=int, default=5)
    parser.add_argument("--num_samples", type=int, default=32)
    parser.add_argument("--keep_last_layer", type=lambda x: x.lower() in ["true", "1", "yes"], default=True)
    parser.add_argument("--num_integrate_batch_size", type=int, default=2)
    parser.add_argument("--mi_downsample_enable", type=lambda x: x.lower() in ["true", "1", "yes"], default=False)
    parser.add_argument("--mi_estimator_iter_num", type=int, default=2)
    parser.add_argument("--mi_estimator_lr", type=float, default=1e-4)
    parser.add_argument("--mi_xz_ratio", type=float, default=0.001)
    parser.add_argument("--mi_yz_ratio", type=float, default=0.001)
    parser.add_argument("--mi_estimator_hidden_dim", type=int, default=128)
    parser.add_argument("--use_residual", type=lambda x: x.lower() in ["true", "1", "yes"], default=True)

    args = parser.parse_args()
    if args.output_dir is None:
        save_path = Path("outputs/train_ckpts_gen") / args.experiment_name
        save_path = save_path / (
            f"Budget-{args.privacy_budget}_LST-{args.lst_enable}_"
            f"Reduce-{args.lst_reduce_factor}_Skip-{args.lst_skip}_"
            f"AutoSkip-{args.auto_skip}_MiDownsample-{args.mi_downsample_enable}"
        )
        args.output_dir = save_path.as_posix()
    return args


if __name__ == "__main__":
    train(parse_args())
