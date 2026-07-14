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

from baselines.snd.modeling import DenoiseModel, DenoiseModelLlama
from baselines.snd.data import MixDatasetLoad, MixDatasetDump, ensure_padding_token, load_dataset
import argparse

setup_seed(12399)


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--privacy_budget", type=int, default=1000)
    arg_parser.add_argument("--model_name", type=str, default="Qwen2-1.5B-Instruct")
    arg_parser.add_argument("--model_type", choices=["qwen2", "llama"], default="qwen2")
    arg_parser.add_argument("--num_train_epochs", type=int, default=1)
    arg_parser.add_argument("--learning_rate", type=float, default=1e-4)
    arg_parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    arg_parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    arg_parser.add_argument("--lora_r", type=int, default=64)
    arg_parser.add_argument("--lr_scheduler_type", type=str, default="constant")
    arg_parser.add_argument("--output_dir", type=str, default=None)
    args = arg_parser.parse_args()

    privacy_budget = args.privacy_budget

    if args.model_type == "qwen2":
        model_name = args.model_name
        denoise_model_cls = DenoiseModel
        save_dir = "denoise_model"
        tokenizer = ensure_padding_token(AutoTokenizer.from_pretrained(model_name))
        model = denoise_model_cls.from_pretrained(
            model_name,
            pad_token_id=tokenizer.pad_token_id,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
    elif args.model_type == "llama":
        model_name = args.model_name
        denoise_model_cls = DenoiseModelLlama
        save_dir = "denoise_model_llama"
        tokenizer = ensure_padding_token(AutoTokenizer.from_pretrained(model_name))
        model = denoise_model_cls.from_pretrained(
            model_name,
            pad_token_id=tokenizer.pad_token_id,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    save_strategy = "epoch"
    save_steps = 100
    evaluation_strategy=None
    eval_steps = 100

    per_device_train_batch_size = args.per_device_train_batch_size
    per_device_eval_batch_size = args.per_device_eval_batch_size
    use_cpu = False

    num_train_epochs = args.num_train_epochs
    learning_rate = args.learning_rate

    lora_r = args.lora_r
    
    train(
        model,
        tokenizer,

        privacy_budget=privacy_budget,
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
            else Path(__file__).parent / save_dir / f"{privacy_budget}"
        ),
        lr_scheduler_type=args.lr_scheduler_type,
        model_type=args.model_type,
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
    privacy_budget,
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
    model_type="qwen2",
):

    dataset = load_dataset(privacy_budget, tokenizer, model_type=model_type)

    

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

    model = get_peft_model(model, lora_config).cuda()

    print(model)


    lr_scheduler_kwargs = {}

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
        lr_scheduler_type=lr_scheduler_type,
        lr_scheduler_kwargs=lr_scheduler_kwargs,
    )

    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()


if __name__ == "__main__":
    main()
