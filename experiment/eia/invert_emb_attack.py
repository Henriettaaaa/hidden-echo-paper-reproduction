import argparse
import gc
import json
import os
from pathlib import Path


os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from tqdm import tqdm
import datasets
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification
import transformers
from utils.noise import get_noisy_embedding
from utils.dataset import get_glue_dataset
import torch.utils.data
import numpy as np
import evaluate
from torch import nn

from baselines.gan.gan_modeling import Generator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/data/songhanlin/models/Qwen2-1.5B-Instruct")
    parser.add_argument("--model_type", type=str, default="qwen2-1.5b")
    parser.add_argument("--dataset_name", type=str, default="financial_phrasebank")
    parser.add_argument("--financial_phrasebank_config", type=str, default="sentences_allagree")
    parser.add_argument("--privacy_budgets", type=int, nargs="+", default=[100, 1000, 5000, 6000])
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--is_gan", action="store_true")
    parser.add_argument("--generator_dir", type=str, default=None)
    parser.add_argument("--generator_dir_template", type=str, default=None)
    parser.add_argument("--generator_checkpoint_template", type=str, default=None)
    parser.add_argument("--generator_epoch", type=int, default=4)
    parser.add_argument("--output_json", type=str, default=None)
    return parser.parse_args()


args = parse_args()
p = args.model_path
model_type = args.model_type
dataset_name = args.dataset_name
privacy_budgets = args.privacy_budgets

tokenizer = AutoTokenizer.from_pretrained(p)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
model = AutoModelForSequenceClassification.from_pretrained(p, torch_dtype=torch.bfloat16)


if dataset_name == "mrpc":
    subset = "mrpc"
    ds, dataset_metrics = get_glue_dataset(subset)
    train_datasets = ds["train"]
    val_datasets = ds["validation"]
    max_length = 128
    text_keys = ["sentence1", "sentence2"]
        
elif dataset_name == "financial_phrasebank":
    ds = datasets.load_dataset('financial_phrasebank', args.financial_phrasebank_config, revision='main')
    ds = ds['train'].train_test_split(test_size=0.1, seed=123, stratify_by_column="label")

    train_datasets = ds["train"]
    val_datasets = ds["test"]
    max_length = 96
    text_keys = ["sentence"]

elif dataset_name == "SetFit/bbc-news":
    ds = datasets.load_dataset('SetFit/bbc-news')

    train_datasets = ds["train"]
    val_datasets = ds["test"]
    max_length = 512
    text_keys = ["text"]

elif dataset_name == "dailymail":
    ds = datasets.load_dataset('dataset/cnn_dailymail_short')

    train_datasets = ds["train"]
    val_datasets = ds["validation"]

    input_field = "article"
    label_field = "highlights"
    
    input_field_len = 256
    label_field_len = 96
    
    max_length = input_field_len
    text_keys = [input_field]
elif dataset_name == "samsum":
    ds = datasets.load_dataset('dataset/samsum')

    train_datasets = ds["train"]
    val_datasets = ds["validation"]

    input_field = "dialogue"
    label_field = "summary"
    
    input_field_len = 80
    label_field_len = 50

    max_length = input_field_len
    text_keys = [input_field]
elif dataset_name == "fr2en":
    ds = datasets.Dataset.from_csv("dataset/damo_mt_testsets_fr2en_iwslt14.csv", column_names=["en", "fr"],)
    ds = ds.train_test_split(test_size=0.1, seed=123)
    train_datasets = ds["train"]
    val_datasets = ds["test"]
    
    input_field = "en"
    label_field = "fr"
    
    input_field_len = 128
    label_field_len = 128
    
    text_keys = [input_field]
    max_length = input_field_len
else:
    raise ValueError(f"Unknown dataset name: {dataset_name}")


embed_layer = nn.Embedding.from_pretrained(model.get_input_embeddings().weight).cuda()
inverse_embed_layer = torch.pinverse(embed_layer.weight.float())

def preprocess_function(examples):
    return tokenizer(
        *[examples[key] for key in text_keys],
        truncation=True,
        padding="max_length",
        return_tensors="pt",
        max_length=max_length,
    )

rouge_metric = evaluate.load("rouge")


def resolve_generator_checkpoint(budget):
    if args.generator_checkpoint_template is not None:
        return args.generator_checkpoint_template.replace("{eta}", str(budget)).replace("{budget}", str(budget))
    if args.generator_dir_template is not None:
        generator_dir = args.generator_dir_template.replace("{eta}", str(budget)).replace("{budget}", str(budget))
        return str(Path(generator_dir) / f"epoch_{args.generator_epoch}" / "generator.pth")
    if args.generator_dir is not None:
        return str(Path(args.generator_dir) / f"epoch_{args.generator_epoch}" / "generator.pth")
    raise ValueError("GAN EIA requires --generator_dir, --generator_dir_template, or --generator_checkpoint_template")


def load_generator_for_budget(budget):
    checkpoint_path = resolve_generator_checkpoint(budget)
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Generator checkpoint not found for budget {budget}: {checkpoint_path}")
    print(f"loading_generator budget={budget} checkpoint={checkpoint_path}")
    return Generator.from_pretrained(
        checkpoint_path,
        model.config.hidden_size,
        model.config.hidden_size,
        2,
    ).to(torch.bfloat16).eval().cuda()


def m(examples, budget, generator_model=None):
    texts = zip(*[examples[key] for key in text_keys])
    texts = [' '.join(t) for t in texts]
    
    input = preprocess_function(examples)
    input_ids = input['input_ids'].cuda()
    attention_mask = input['attention_mask'].cuda()
    
    inputs_embeds = embed_layer(input_ids)
    
    noisy_inputs_embeds, _ = get_noisy_embedding(inputs_embeds, budget, True, model_type=model_type)
    if generator_model is not None:
        noisy_inputs_embeds = generator_model(noisy_inputs_embeds, None)
    invert = noisy_inputs_embeds.float() @ inverse_embed_layer
    invert_tokens = invert.argmax(dim=-1)
    invert_text = tokenizer.batch_decode(invert_tokens)

    score = rouge_metric.compute(predictions=invert_text, references=texts)
    return score['rouge1']

total_scores = {}

batch_size = args.batch_size
num_samples = min(args.num_samples, len(train_datasets))
with torch.no_grad():
    for budget in privacy_budgets:
        generator_model = load_generator_for_budget(budget) if args.is_gan else None
        total_scores[budget] = []
        for i in tqdm(range(0, num_samples, batch_size), desc=f"budget={budget}"):
            examples = train_datasets[i:i+batch_size]
            score = m(examples, budget, generator_model=generator_model)
            total_scores[budget].append(score)
        if generator_model is not None:
            del generator_model
            gc.collect()
            torch.cuda.empty_cache()

results = {}
for budget in privacy_budgets:
    rouge1 = float(np.mean(total_scores[budget]))
    ep = float(1 - rouge1)
    print(f"Budget: {budget}")
    print(ep)
    results[str(budget)] = {
        "rouge1": rouge1,
        "ep": ep,
    }

if args.output_json is not None:
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_path": args.model_path,
        "model_type": args.model_type,
        "dataset_name": args.dataset_name,
        "financial_phrasebank_config": args.financial_phrasebank_config,
        "is_gan": args.is_gan,
        "generator_epoch": args.generator_epoch if args.is_gan else None,
        "privacy_budgets": privacy_budgets,
        "batch_size": args.batch_size,
        "num_samples": num_samples,
        "scores": results,
    }
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"saved_json: {output_json}")
