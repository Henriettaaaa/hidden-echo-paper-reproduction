import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from matplotlib import pyplot as plt
# import peft
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

import gc

from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from utils.model import (
    setup_seed,
    get_base_classification_model_for_training,
    parse_args_for_model_train_options,
    load_classification_model_checkpoint,
)
from utils.dataset import get_glue_dataset

from modeling.my.configuration import AdditionalConfig
from modeling.my.split import SplittedQwen2ForSequenceClassification
from torch.utils.data import DataLoader
import datasets
import torch.nn.functional
import torch.utils.data
import json
from tqdm import tqdm

setup_seed(12399)


def evaluate_model(model, dataloader):
    with torch.no_grad():
        auc = 0
        for batch in tqdm(dataloader):
            inputs = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            labels = batch["labels"]
            outputs = model(input_ids=inputs, attention_mask=attention_mask)
            logits = outputs.logits
            
            num_labels = logits.shape[-1]
            if num_labels > 2:
                probabilities = torch.nn.functional.softmax(logits.float(), dim=-1).cpu().numpy()
                auc += sklearn.metrics.roc_auc_score(labels, probabilities, multi_class="ovr")
            else:
                auc += sklearn.metrics.roc_auc_score(labels, logits[:, 1].float().cpu().numpy())
        
        return auc / len(dataloader)


def main():
    dataset_name = "mrpc"
    model_name = "path/to/model"
    checkpoint_path = "path/to/checkpoint"
    
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
                max_length=256,
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
                max_length=768,
            )

    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    
    model = load_classification_model_checkpoint(
        checkpoint_path,
        model_path=model_name,
        num_labels=num_labels,
        model_cls=SplittedQwen2ForSequenceClassification,
    )
    model = model.cuda().eval()
    
    valid_dataset = val_datasets.map(preprocess_function, batched=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False, collate_fn=transformers.default_data_collator)

    final_auc = 0
    for _ in range(20):
        auc = evaluate_model(model, valid_dataloader)
        final_auc += auc
    final_auc /= 20
    print(f"Final AUC: {final_auc}")
    
    
if __name__ == "__main__":
    main()