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

from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from utils.model import (
    setup_seed,
    get_base_classification_model_for_training,
)
from utils.dataset import get_glue_dataset
from utils.noise import get_noisy_embedding

from torch.utils.data import DataLoader
import datasets
import torch.nn.functional
import torch.utils.data
import json
from torch import nn
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
    
    
    args = parser.parse_args()
    
    

    save_strategy = "epoch"
    save_steps = 100
    # save_strategy = "steps"
    evaluation_strategy="epoch"
    # evaluation_strategy=None
    # evaluation_strategy = "steps"
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
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model_config = model.config
    
    
    generator_checkpoint_dirname = ""
    if "t5" in model_name:
        generator_checkpoint_dirname = "gan_models_t5"
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    generator_checkpoint_path = Path(__file__).parent / generator_checkpoint_dirname / f"{dataset_name}_privacy_{privacy_budget}/epoch_{generator_epoch}/generator.pth"
    generator = Generator.from_pretrained(generator_checkpoint_path, model_config.hidden_size, model_config.hidden_size, 2).to(torch.bfloat16).eval().cuda()

    if "t5" in model_name:
        llm_type = "t5-large"
    else:
        raise ValueError("Unknown LLM model")



    model_embed_layer = model.get_input_embeddings()
    
    class WrappedEmbedLayer(nn.Module):
        def forward(self, input_ids):
            inputs_embeds = model_embed_layer(input_ids)
            noisy_embeds, _ = get_noisy_embedding(inputs_embeds, privacy_budget, True, model_type=llm_type)
            with torch.no_grad():
                generated_embeds = generator(noisy_embeds, None)
            return generated_embeds
    
    model.set_input_embeddings(WrappedEmbedLayer())


    save_dirname = ""
    if "t5" in model_name:
        save_dirname = "task_models_t5"
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
        save_path=Path(__file__).parent / save_dirname / f"{dataset_name}_privacy_{privacy_budget}",
        lr_scheduler_type="constant",
    )


# ######################################
# ######################################
# ######################################
# ######################################


class MyTrainer(Seq2SeqTrainer):
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
):


    if dataset_name == "dailymail":
        ds = datasets.load_dataset('cnn_dailymail_short')

        train_datasets = ds["train"]
        val_datasets = ds["validation"]

        input_field = "article"
        label_field = "highlights"
        
        input_field_len = 256
        label_field_len = 96
        
        prefix = "summarize: "
    elif dataset_name == "samsum":
        ds = datasets.load_dataset('dataset/samsum')

        train_datasets = ds["train"]
        val_datasets = ds["validation"]

        input_field = "dialogue"
        label_field = "summary"
        
        input_field_len = 80
        label_field_len = 50
        
        prefix = "summarize: "
    elif dataset_name == "fr2en":
        ds = datasets.Dataset.from_csv("dataset/damo_mt_testsets_fr2en_iwslt14.csv", column_names=["en", "fr"],)
        ds = ds.train_test_split(test_size=0.1, seed=123)
        train_datasets = ds["train"]
        val_datasets = ds["test"]
        
        input_field = "en"
        label_field = "fr"
        
        input_field_len = 128
        label_field_len = 128
        
        prefix = "translate English to French: "
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    def preprocess_function(examples):
        batch_inputs, batch_targets = examples[input_field], examples[label_field]
        batch_inputs = [prefix + inp for inp in batch_inputs]
        batch_data = tokenizer(
            batch_inputs, 
            padding=True, 
            max_length=input_field_len,
            truncation=True, 
            return_tensors="pt"
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch_targets, 
                padding=True, 
                max_length=label_field_len,
                truncation=True, 
                return_tensors="pt"
            )["input_ids"]

            # batch_data['decoder_input_ids'] = model.prepare_decoder_input_ids_from_labels(labels)
            end_token_index = torch.where(labels == tokenizer.eos_token_id)[1]
            for idx, end_idx in enumerate(end_token_index):
                labels[idx][end_idx+1:] = -100
            batch_data['labels'] = labels
        return batch_data
        


    tokenized_train_dataset = train_datasets.map(preprocess_function, batched=True)
    tokenized_val_dataset = val_datasets.map(preprocess_function, batched=True)


    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=lora_r,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=[
            "q",
            "v",
        ],
        bias="none",
        layers_pattern="block",
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

    training_args = Seq2SeqTrainingArguments(
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
        # predict_with_generate=True,
        # generation_max_length=64,
        # metric_for_best_model="bleu",
        # greater_is_better=True,
        # load_best_model_at_end=True,
    )

    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        # compute_metrics=compute_metrics,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, padding=True,
            label_pad_token_id=-100,
        ),
    )


    try:
        trainer.train()
    except KeyboardInterrupt:
        pass
    eval_results = {}

    trained_model = trainer.model            

    data_loader = DataLoader(tokenized_val_dataset.remove_columns(val_datasets.column_names), 
                                batch_size=per_device_eval_batch_size, 
                                collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, label_pad_token_id=-100)
                                )
    for _ in range(5):
        bleu_metric = evaluate.load("sacrebleu")
        rouge_metric = evaluate.load("rouge")
        for test_batch in tqdm(data_loader):
            input_ids = test_batch["input_ids"].cuda()
            attention_mask = test_batch["attention_mask"].cuda()
            
            generate_tokens = trained_model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=64)
            # input_text = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            generate_text = tokenizer.batch_decode(generate_tokens, skip_special_tokens=True)
            
            label_ids = test_batch["labels"]
            label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)
            reference = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
            # reference = label_text
            print(list(zip(generate_text,reference )))
            bleu_metric.add_batch(predictions=generate_text, references=reference)
            rouge_metric.add_batch(predictions=generate_text, references=reference)

        eval_results.setdefault("bleu", [])
        eval_results["bleu"].append(bleu_metric.compute()['score'])
        for k,v in rouge_metric.compute().items():
            eval_results.setdefault(k, [])
            eval_results[k].append(v)
    
    avg_results = {k: np.mean(v) for k,v in eval_results.items()}
    print(avg_results)
    with open(f"{save_path}/eval_results.json", "w") as f:
        json.dump(avg_results, f)

main()
