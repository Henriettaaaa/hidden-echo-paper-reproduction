import os
from pathlib import Path

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
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
# from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

import gc

from transformers import AutoModelForSequenceClassification
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from utils.model import (
    setup_seed,
    get_base_classification_model_for_training,
    parse_args_for_model_train_options,
)
from utils.dataset import get_glue_dataset
from utils.noise import get_noisy_embedding

from modeling.my.configuration import AdditionalConfig
from modeling.my.split import SplittedQwen2ForSequenceClassification
from torch.utils.data import DataLoader
import datasets
import torch.nn.functional
import torch.utils.data
import json

from baselines.gan.gan_modeling import Generator

setup_seed(12399)

import argparse

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mrpc', help='Name of dataset')
    parser.add_argument('--privacy_budget', type=int, default=100, help='Privacy budget')
    parser.add_argument('--generator_epoch', type=int, default=4, help='Generator epoch')
    parser.add_argument('--model_name', type=str,  help='Model name')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--per_device_train_batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_train_epochs', type=int, default=15, help='Number of epochs for training model')
    parser.add_argument(
        "--financial_phrasebank_config",
        type=str,
        default="sentences_50agree",
        choices=["sentences_50agree", "sentences_allagree"],
    )
    parser.add_argument("--lr_scheduler_type", type=str, default="constant")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--generator_dir", type=str, default=None)
    
    
    args = parser.parse_args()
    
    
    datasets2labels = {
        "mrpc": 2,
        "financial_phrasebank": 3,
        "bbc-news": 5,
    }
    

    save_strategy = "epoch"
    save_steps = 100
    evaluation_strategy="epoch"
    eval_steps = 100

    use_cpu = False

    lora_r = 16


    dataset_name = args.dataset
    privacy_budget = args.privacy_budget
    generator_epoch = args.generator_epoch
    model_name = args.model_name
    learning_rate = args.learning_rate
    per_device_train_batch_size = args.per_device_train_batch_size
    per_device_eval_batch_size = args.per_device_eval_batch_size
    num_train_epochs = args.num_train_epochs

    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=datasets2labels[dataset_name],
        pad_token_id=tokenizer.pad_token_id,
        torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model_config = model.config
    
    
    generator_checkpoint_dirname = ""
    if "Qwen2" in model_name:
        generator_checkpoint_dirname = "gan_models"
    elif "Llama" in model_name:
        generator_checkpoint_dirname = "gan_models_llama"
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    generator_checkpoint_path = (
        Path(args.generator_dir) / f"epoch_{generator_epoch}" / "generator.pth"
        if args.generator_dir is not None
        else Path(__file__).parent / generator_checkpoint_dirname / f"{dataset_name}_privacy_{privacy_budget}/epoch_{generator_epoch}/generator.pth"
    )
    generator = Generator.from_pretrained(generator_checkpoint_path, model_config.hidden_size, model_config.hidden_size, 2).to(torch.bfloat16).eval().cuda()


    if "Qwen" in model_name:
        llm_type = "qwen2-1.5b"
    elif "Llama" in model_name:
        llm_type = "llama-3.2-1b"
    else:
        raise ValueError("Unknown LLM model")


    model_foward = model.forward
    def new_forward(self, input_ids, attention_mask, **kwargs):
        inputs_embeds = self.get_input_embeddings()(input_ids)
        noisy_embeds, _ = get_noisy_embedding(inputs_embeds, privacy_budget, True, model_type=llm_type)
        with torch.no_grad():
            generated_embeds = generator(noisy_embeds, None)
        kwargs.pop("inputs_embeds", None)
        kwargs.pop("input_ids", None)
        kwargs.pop("attention_mask", None)
        return model_foward(None, inputs_embeds=generated_embeds, attention_mask=attention_mask, **kwargs)

    model.forward = new_forward.__get__(model)


    save_dirname = ""
    if "Qwen2" in model_name:
        save_dirname = "task_models2"
    elif "Llama" in model_name:
        save_dirname = "task_models_llama"
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    train(
        model,
        tokenizer,

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
        save_path=(
            Path(args.output_dir)
            if args.output_dir is not None
            else Path(__file__).parent / save_dirname / f"{dataset_name}_privacy_{privacy_budget}"
        ),
        lr_scheduler_type=args.lr_scheduler_type,
        financial_phrasebank_config=args.financial_phrasebank_config,
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
    model,
    tokenizer,
    *,
    dataset_name="mrpc",
    num_train_epochs=4,
    save_strategy="steps",
    save_steps=2000,
    evaluation_strategy="steps",
    eval_steps=2000,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=64,
    warmup_steps=50,
    # weight_decay=0.01,
    weight_decay=0,
    learning_rate=5e-5,
    logging_steps=10,
    use_cpu=False,
    lora_r=64,
    save_path="./ckpt",
    lr_scheduler_type="warmup_stable_decay",
    financial_phrasebank_config="sentences_50agree",
):

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
        
        ds = datasets.load_dataset(
            "financial_phrasebank",
            financial_phrasebank_config,
            revision="main",
        )["train"]
        if financial_phrasebank_config == "sentences_allagree":
            expected_size = 1811 + 226 + 227
            if len(ds) != expected_size:
                raise ValueError(
                    f"Expected financial_phrasebank/{financial_phrasebank_config} "
                    f"to contain {expected_size} rows, got {len(ds)}"
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
            train_datasets = split["train"]
            val_datasets = dev_test["train"]
            test_datasets = dev_test["test"]
        else:
            ds = ds.train_test_split(test_size=0.1, seed=123, stratify_by_column="label")
            train_datasets = ds["train"]
            val_datasets = ds["test"]
            test_datasets = ds["test"]
        def preprocess_function(examples):
            return tokenizer(
                examples["sentence"],
                truncation=True,
                padding="max_length",
                return_tensors="pt",
                max_length=96,
            )

    elif dataset_name == "bbc-news":
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

    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    if dataset_name != "financial_phrasebank":
        test_datasets = val_datasets


    tokenized_train_dataset = train_datasets.map(preprocess_function, batched=True)
    tokenized_val_dataset = val_datasets.map(preprocess_function, batched=True)
    tokenized_test_dataset = test_datasets.map(preprocess_function, batched=True)


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
        layers_pattern="layers",
    )

    model = get_peft_model(model, lora_config)

    print(model)


    steps_per_epoch = len(tokenized_train_dataset) // per_device_train_batch_size
    total_steps = steps_per_epoch * num_train_epochs

    lr_scheduler_kwargs = {}

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
        save_total_limit=4,  
        use_cpu=use_cpu,
        seed=123,
        # lr_scheduler_type="constant_with_warmup",
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
