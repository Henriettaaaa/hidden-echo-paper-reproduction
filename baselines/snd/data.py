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
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
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
from baselines.snd.modeling import Qwen2ForSequenceClassification, LlamaForSequenceClassification
from torch.utils.data import DataLoader
import datasets
import torch.nn.functional
import torch.utils.data
import json
from pathlib import Path
import datasets

class MixDatasetDump(torch.utils.data.Dataset):
    def __init__(self, model, tokenizer, privacy_budget):
        
        dataset_options = [
            ("tweet_eval", "sentiment", "train", 5000, "text"),
            ("tweet_eval", "offensive", "train", 5000, "text"),
            ("wikitext", "wikitext-2-v1", "train", 5000, "text"),
        ]
        
        
        
        
        
        
        datasets_list = []
        for dataset, name, split, limit, text_key in dataset_options:
            dataset = datasets.load_dataset(dataset, name)[split]
            if limit is not None:
                dataset = dataset.select(range(limit))
            dataset = dataset.select_columns([text_key])
            datasets_list.append(dataset)
        
        cat_datasets = datasets.concatenate_datasets(datasets_list)
        
        self.datasets = cat_datasets
        self.length = len(cat_datasets)
        
        self.model = model
        self.tokenizer = tokenizer
        self.privacy_budget = privacy_budget

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sample = self.datasets[idx]
        return sample
        
    def collate_fn(self, batch):
        
        texts = [sample["text"] for sample in batch]
        
        inputs = self.tokenizer(texts, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
        with torch.no_grad():
            model_output = self.model(input_ids=inputs["input_ids"].cuda(), attention_mask=inputs["attention_mask"].cuda(), privacy_budget=self.privacy_budget)
        cls_embed = model_output.cls_embeds
        noise = model_output.noise
        
        if self.privacy_budget == 0:
            return {
                "text": texts,
                "cls_embed": cls_embed.cpu(),
            }
        else:
            return {
                "text": texts,
                "cls_embed": cls_embed.cpu(),
                "noise": noise.cpu(),
            }
    


class MixDatasetLoad(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, clean_dataset_dir, tokenizer):
        self.tokenizer = tokenizer
        
        self.dataset_dir = Path(dataset_dir)
        self.files = list(self.dataset_dir.glob("chunk_*.pt"))
        
        self.clean_dataset_dir = Path(clean_dataset_dir)
        self.clean_files = list(self.clean_dataset_dir.glob("chunk_*.pt"))
        
        self.current_file_index = 0
        self.current_data = torch.load(self.files[self.current_file_index], map_location="cpu")
        self.current_clean_data = torch.load(self.clean_files[self.current_file_index], map_location="cpu")


    def __len__(self):
        return len(self.files) * len(self.current_data)

    def __getitem__(self, idx):
        target_chunk_index = idx // len(self.current_data)
        target_data_index = idx % len(self.current_data)
        
        if target_chunk_index != self.current_file_index:
            self.current_file_index = target_chunk_index
            self.current_data = torch.load(self.files[self.current_file_index], map_location="cpu")
            self.current_clean_data = torch.load(self.clean_files[self.current_file_index], map_location="cpu")
        
        data = self.current_data[target_data_index]
        clean_data = self.current_clean_data[target_data_index]
        
        assert data["text"] == clean_data["text"]
        
        inputs = self.tokenizer(data["text"], return_tensors="pt", padding="max_length", truncation=True, max_length=512)

        res =  {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "noisy_cls_embed": data["cls_embed"].squeeze(),
            "clean_cls_embed": clean_data["cls_embed"].squeeze(),
            "noise": data["noise"].squeeze(),
        }
        
        
        
        
        return res
        
def gen_dataset(model, tokenizer, privacy_budgets):
    
    chunk_size = 2048000
    
    for budget in privacy_budgets:
        dataset = MixDatasetDump(model, tokenizer, budget)
        dataset_dir = Path(__file__).parent / "mixed_datasets_llama" / f"{budget}"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        data_loader = DataLoader(dataset, batch_size=64, collate_fn=dataset.collate_fn, shuffle=False)
        
        chunk = []
        chunk_index = 0
        for data in tqdm(data_loader):
            for j in range(len(data["text"])):
                chunk.append({
                    k: v[j] for k, v in data.items()
                })

            if len(chunk) >= chunk_size:
                torch.save(chunk, dataset_dir / f"chunk_{chunk_index}.pt")
                chunk_index += 1
                chunk = []
                
        if len(chunk) > 0:
            torch.save(chunk, dataset_dir / f"chunk_{chunk_index}.pt")
        print(f"Finished dumping budget {budget}")
        
    print("Finished dumping all budgets")


def load_dataset(privacy_budget, tokenizer, model_type="qwen2"):
    if model_type == "qwen2":
        dataset_dir = Path(__file__).parent / "mixed_datasets" / f"{privacy_budget}"
        clean_dataset_dir = Path(__file__).parent / "mixed_datasets" / f"0"
    elif model_type == "llama":
        dataset_dir = Path(__file__).parent / "mixed_datasets_llama" / f"{privacy_budget}"
        clean_dataset_dir = Path(__file__).parent / "mixed_datasets_llama" / "0"
    loaded_dataset = MixDatasetLoad(dataset_dir, clean_dataset_dir, tokenizer)
    # print(f"Loaded dataset length: {len(loaded_dataset)}")
    # print(f"First item: {loaded_dataset[0]}")
    return loaded_dataset

if __name__ == "__main__":
    # model_type = "qwen2"
    model_type = "llama"
    if model_type == "qwen2":
        model_name = "Qwen2-1.5B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = Qwen2ForSequenceClassification.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda")
        model.eval()
        budgets = [0, 100, 1000, 5000, 6000]
    elif model_type == "llama":
        model_name = "Llama-3.2-1B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model = LlamaForSequenceClassification.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda", pad_token_id=tokenizer.pad_token_id)
        model.eval()
        budgets = [0, 1000, 4000, 5000] 
    gen_dataset(model, tokenizer, budgets)
