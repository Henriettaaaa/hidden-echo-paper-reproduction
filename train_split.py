import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import evaluate
import numpy as np
from matplotlib import pyplot as plt
import peft
import sklearn
import sklearn.metrics
import transformers
from typing_extensions import override
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

import gc

from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from utils.model import (
    setup_seed,
    get_base_classification_model_for_training,
    parse_args_for_model_train_options,
)
from utils.dataset import get_glue_dataset

from modeling.my.configuration import AdditionalConfig
from modeling.my.split import SplittedQwen2ForSequenceClassification
from torch.utils.data import DataLoader
import datasets
import torch.nn.functional
import torch.utils.data
import json

setup_seed(12399)


def main():
    args = parse_args_for_model_train_options()
    print(args)
    print(args.model_options.__dict__)

    experiment_name = args.experiment_name

    dataset_name = args.dataset_name

    model_name = args.model_path
    model_max_len = args.max_len

    save_strategy = "epoch"
    save_steps = 100
    # save_strategy = "steps"
    evaluation_strategy="epoch"
    # evaluation_strategy=None
    # evaluation_strategy = "steps"
    eval_steps = 100

    per_device_train_batch_size = args.train_batch_size
    # per_device_train_batch_size=4
    per_device_eval_batch_size = args.eval_batch_size
    use_cpu = False

    num_train_epochs = args.num_train_epochs
    learning_rate = args.learning_rate

    lora_r = args.lora_rank

    custom_model_config = args.model_options

    names = {
        "privacy_budget": "Budget",
        "lst_backbone_use_lora": "Lora",
        "lst_enable": "LST",
        "lst_reduce_factor": "Reduce",
        "lst_input_type": "InputType",
        "lst_skip": "Skip",
        "lst_random_init": "RandomInit",

        "auto_skip": "AutoSkip",
        "num_reserved_layers": "ReservedLayers",
        "num_integrate_step": "IntegrateStep",
        "num_samples": "Samples",
        "keep_last_layer": "KeepLastLayer",
        
        "mi_downsample_enable": "MiDownsample",
        "mi_estimator_iter_num": "IterNum",
        "mi_estimator_lr": "Lr",
        "mi_xz_ratio": "XzRatio",
        "mi_yz_ratio": "YzRatio",
        "mi_estimator_hidden_dim": "HiddenDim",
    }

    name_value_map = {
        "lst_skip": lambda x: x.step if isinstance(x, range) else x,
    }
    
    show_if = {
        "lst_reduce_factor": lambda opt: opt.lst_enable,
        "lst_skip": lambda opt: opt.lst_enable,
        "lst_temperature": lambda opt: opt.lst_enable,
        "lst_input_type": lambda opt: opt.lst_enable,
        "lst_random_init": lambda opt: opt.lst_enable,

        "auto_skip": lambda opt: opt.lst_enable,
        "num_reserved_layers": lambda opt: opt.lst_enable and opt.auto_skip,
        "num_integrate_step": lambda opt: opt.lst_enable and opt.auto_skip,
        "num_samples": lambda opt: opt.lst_enable and opt.auto_skip,
        "keep_last_layer": lambda opt: opt.lst_enable and opt.auto_skip,
        
        "mi_downsample_enable": lambda opt: opt.lst_enable,
        "mi_estimator_iter_num": lambda opt: opt.lst_enable and opt.mi_downsample_enable,
        "mi_estimator_lr": lambda opt: opt.lst_enable and opt.mi_downsample_enable,
        "mi_xz_ratio": lambda opt: opt.lst_enable and opt.mi_downsample_enable,
        "mi_yz_ratio": lambda opt: opt.lst_enable and opt.mi_downsample_enable,
        "mi_estimator_hidden_dim": lambda opt: opt.lst_enable and opt.mi_downsample_enable,
    }

    save_path = f"./outputs/train_ckpts/{experiment_name}/"
    for k, v in custom_model_config.__dict__.items():
        if k in names:
            if k in show_if and not show_if[k](custom_model_config):
                continue
            if k in name_value_map:
                v = name_value_map[k](v)
            save_path += f"{names[k]}-{v}_"

    custom_model_config.lst_skip = list(custom_model_config.lst_skip or [])

    train(
        custom_config=custom_model_config,
        model_name=model_name,
        model_max_len=model_max_len,
        dataset_name=dataset_name,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        save_steps=save_steps if save_strategy == "steps" else None,
        save_strategy=save_strategy,
        evaluation_strategy=evaluation_strategy,
        eval_steps=eval_steps if evaluation_strategy == "steps" else None,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        use_cpu=use_cpu,
        lora_r=lora_r,
        save_path=save_path,
        lr_scheduler_type=args.lr_scheduler_type,
        lr_scheduler_wsd_decay_epochs=args.lr_scheduler_wsd_decay_epochs,
    )


# ######################################
# ######################################
# ######################################
# ######################################


class MyTrainer(Trainer):
    @override
    def save_model(self, output_dir=None, _internal_call=False):
        self.model.config.save_pretrained(output_dir)

        
        for metric in ["eval_accuracy", "eval_f1", "eval_mcc", "eval_auc"]:
            if not any(metric in log for log in self.state.log_history):
                continue
            eval_logs = [log for log in self.state.log_history if metric in log]
            x = [log["epoch"] for log in eval_logs]
            y = [log[metric] for log in eval_logs]
            plt.plot(x, y)
            plt.xlabel("Epoch")
            plt.ylabel(metric)
            plt.title(f"{metric} Curve")
            plt.grid(True)
            plt.savefig(f"{output_dir}/{metric}_curve.png")
            plt.close()

        
        train_logs = [log for log in self.state.log_history if "loss" in log]
        x = [log["epoch"] for log in train_logs]
        y = [log["loss"] for log in train_logs]
        plt.plot(x, y, label="train_loss")
        eval_logs = [log for log in self.state.log_history if "eval_loss" in log]
        x = [log["epoch"] for log in eval_logs]
        y = [log["eval_loss"] for log in eval_logs]
        plt.plot(x, y, label="eval_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Loss Curve")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_dir}/loss_curve.png")
        plt.close()

        return super().save_model(output_dir, _internal_call)



def train(
    *,
    custom_config: AdditionalConfig,
    model_name="Qwen2-1.5B-Instruct",
    model_max_len=512,
    dataset_name="mrpc",
    num_train_epochs=4,
    save_strategy="steps",
    save_steps=2000,
    evaluation_strategy="steps",
    eval_steps=2000,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=64,
    warmup_steps=50,
    
    weight_decay=0,
    learning_rate=5e-5,
    logging_steps=10,
    use_cpu=False,
    lora_r=64,
    save_path="./ckpt",
    lr_scheduler_type="warmup_stable_decay",
    lr_scheduler_wsd_decay_epochs=2,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)


    if dataset_name == "mrpc":
        num_labels = 2
        
        subset = "mrpc"
        ds, dataset_metrics = get_glue_dataset(subset)
        train_datasets = ds["train"]
        val_datasets = ds["validation"]
        def preprocess_function(examples):
            return tokenizer(
                examples["sentence1"],
                examples["sentence2"],
                truncation=True,
                padding="max_length",
                return_tensors="pt",
                max_length=128,
            )
            
    elif dataset_name == "financial_phrasebank":
        num_labels = 3
        
        ds = datasets.load_dataset('financial_phrasebank', 'sentences_50agree', revision='main')
        ds = ds['train'].train_test_split(test_size=0.1, seed=123, stratify_by_column="label")

        train_datasets = ds["train"]
        val_datasets = ds["test"]
        def preprocess_function(examples):
            return tokenizer(
                examples["sentence"],
                truncation=True,
                padding="max_length",
                return_tensors="pt",
                max_length=96,
            )
    
    elif dataset_name == "SetFit/bbc-news":
        num_labels = 5

        ds = datasets.load_dataset('SetFit/bbc-news')

        train_datasets = ds["train"]
        val_datasets = ds["test"]
        def preprocess_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                return_tensors="pt",
                max_length=512,
            )
            
    elif dataset_name == "tweet":
        num_labels = 2
        
        ds = datasets.load_dataset('soda-lmu/tweet-annotation-sensitivity-2')
        
        ds = ds.filter(lambda x: all(v is not None for v in x.values()))
        ds = ds.class_encode_column("offensive_language")
        ds = ds['train'].train_test_split(train_size=2000, seed=123, stratify_by_column="offensive_language")
        ds = ds['train'].train_test_split(train_size=0.8, seed=123, stratify_by_column="offensive_language")
        
        ds = ds.select_columns(["tweet_hashed", "offensive_language", "age", "education"])
        ds = ds.rename_column("tweet_hashed", "text")
        ds = ds.rename_column("offensive_language", "label")
        
        train_datasets = ds["train"]
        val_datasets = ds["test"]
        
        edu2text = {
            1: "Less than high school",
            2: "High school",
            3: "Some college",
            4: "College graduate",
            5: "Master's degree or professional degree",
            6: "Doctoral degree",
        }
        
        def preprocess_function(examples):
            texts = []
            for i in range(len(examples["text"])):
                txt = examples["text"][i]
                age = examples["age"][i]
                education = edu2text[examples["education"][i]]
                txt = f"Age: {age}, Education: {education}, Content: {txt}"
                texts.append(txt)
                
            return tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
                max_length=64,
            )
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")


    model = get_base_classification_model_for_training(
        num_labels=num_labels,
        custom_config=custom_config,
        model_cls=SplittedQwen2ForSequenceClassification,
        model_path=model_name,
    )

    print(model)


    tokenized_train_dataset = train_datasets.map(preprocess_function, batched=True)
    tokenized_val_dataset = val_datasets.map(preprocess_function, batched=True)
    
    ds_split = tokenized_val_dataset.train_test_split(test_size=0.5, seed=123)
    tokenized_val_dataset = ds_split["train"]
    tokenized_test_dataset = ds_split["test"]

    if custom_config.auto_skip:
        if dataset_name == "SetFit/bbc-news":
            sample_datasets = train_datasets.class_encode_column("label").train_test_split(
                test_size=custom_config.num_samples, stratify_by_column="label", seed=123
            )["test"]
        else:
            sample_datasets = train_datasets.train_test_split(
                test_size=custom_config.num_samples, stratify_by_column="label", seed=123
            )["test"]
        tokenized_sample_dataset = sample_datasets.map(
            preprocess_function, batched=True
        )
        sample_dataloader = DataLoader(
            tokenized_sample_dataset,
            batch_size=custom_config.num_integrate_batch_size,
            collate_fn=transformers.default_data_collator,
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
    
    modules_to_save += [
        "client_denoise",
        "server_downsample",
    ]

    print(modules_to_save)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=lora_r,
        lora_alpha=16,
        lora_dropout=0.05,
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

    print(model)


    steps_per_epoch = len(tokenized_train_dataset) // per_device_train_batch_size
    total_steps = steps_per_epoch * num_train_epochs

    lr_scheduler_kwargs = {}
    if lr_scheduler_type == "warmup_stable_decay":
        decay_steps = lr_scheduler_wsd_decay_epochs * steps_per_epoch
        lr_scheduler_kwargs = {
            "num_stable_steps": total_steps - decay_steps,
            "num_decay_steps": decay_steps,
        }


    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        if num_labels > 2:
            probabilities = torch.nn.functional.softmax(torch.tensor(pred.predictions), dim=-1).numpy()
            auc = sklearn.metrics.roc_auc_score(labels, probabilities, multi_class="ovr")
        else:
            auc = sklearn.metrics.roc_auc_score(labels, pred.predictions[:, 1])
        accuracy = sklearn.metrics.accuracy_score(labels, preds)
        if num_labels > 2:
            f1 = sklearn.metrics.f1_score(labels, preds, average="macro")
        else:
            f1 = sklearn.metrics.f1_score(labels, preds)
        return {
            
            "auc": auc,
            "accuracy": accuracy,
            "f1": f1,
        }

    training_args = TrainingArguments(
        output_dir=save_path,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        warmup_steps=warmup_steps,  
        weight_decay=weight_decay,  
        learning_rate=learning_rate,
        logging_dir=save_path,  
        logging_steps=logging_steps,
        evaluation_strategy=evaluation_strategy,
        eval_steps=eval_steps,
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=2,  
        use_cpu=use_cpu,
        seed=123,
        
        lr_scheduler_type=lr_scheduler_type,
        lr_scheduler_kwargs=lr_scheduler_kwargs,
        metric_for_best_model="auc",
        greater_is_better=True,
        load_best_model_at_end=True,
    )

    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        compute_metrics=compute_metrics,
    )


    try:
        trainer.train()
    except KeyboardInterrupt:
        pass

    print(
        f"embedding_data_transferred: {trainer.model.base_model.total_embedding_data_transferred}"
    )
    print(
        f"hiddens_data_transferred: {trainer.model.base_model.total_hidden_states_data_transferred}"
    )
    
    with open(f"{save_path}/data_transfer.txt", "w") as f:
        f.write(f"embedding_data_transferred: {trainer.model.base_model.total_embedding_data_transferred}\n")
        f.write(f"hiddens_data_transferred: {trainer.model.base_model.total_hidden_states_data_transferred}\n")

    eval_results = {}

    for _ in range(5):
        eval_result = trainer.evaluate(tokenized_test_dataset)
        for k, v in eval_result.items():
            if k not in eval_results:
                eval_results[k] = []
            eval_results[k].append(v)
  
    final_eval_results = {}
    for k, v in eval_results.items():
        final_eval_results[k] = sum(v) / len(v)
    with open(f"{save_path}/eval_results.txt", "w") as f:
        f.write(json.dumps(final_eval_results))


main()
