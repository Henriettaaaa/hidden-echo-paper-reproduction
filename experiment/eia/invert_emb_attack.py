import os


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

p = 'Qwen2-1.5B-Instruct'
model_type = 'qwen2-1.5b'
# p = 't5-large'
# model_type = 't5-large'
# p = 'Llama-3.2-1B-Instruct'
# model_type = 'llama-3.2-1b'

is_gan = False


tokenizer = AutoTokenizer.from_pretrained(p)
tokenizer.pad_token_id = tokenizer.eos_token_id
model = AutoModelForSequenceClassification.from_pretrained(p, torch_dtype=torch.bfloat16)

if is_gan:
    generator_model = Generator.from_pretrained(
        "path/to/generator.pth", 
        model.config.hidden_size, model.config.hidden_size, 2
    ).to(torch.bfloat16).cuda()


dataset_name = "mrpc"
# dataset_name = "financial_phrasebank"
# dataset_name = "SetFit/bbc-news"
# dataset_name = "dailymail"
# dataset_name = "samsum"
# dataset_name = "fr2en"

privacy_budgets = [100, 1000, 5000, 6000]
# privacy_budgets = [1000, 4000, 5000]
# privacy_budgets = [20,30,40]


if dataset_name == "mrpc":
    subset = "mrpc"
    ds, dataset_metrics = get_glue_dataset(subset)
    train_datasets = ds["train"]
    val_datasets = ds["validation"]
    max_length = 128
    text_keys = ["sentence1", "sentence2"]
        
elif dataset_name == "financial_phrasebank":
    ds = datasets.load_dataset('financial_phrasebank', 'sentences_50agree', revision='main')
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


def m(examples):
    texts = zip(*[examples[key] for key in text_keys])
    texts = [' '.join(t) for t in texts]
    
    input = preprocess_function(examples)
    input_ids = input['input_ids'].cuda()
    attention_mask = input['attention_mask'].cuda()
    
    inputs_embeds = embed_layer(input_ids)
    
    scores = {}
    
    for budget in privacy_budgets:
        noisy_inputs_embeds, _ = get_noisy_embedding(inputs_embeds, budget, True, model_type=model_type)
        if is_gan:
            noisy_inputs_embeds = generator_model(noisy_inputs_embeds, None)
        invert = noisy_inputs_embeds.float() @ inverse_embed_layer
        invert_tokens = invert.argmax(dim=-1)
        invert_text = tokenizer.batch_decode(invert_tokens)

        score = rouge_metric.compute(predictions=invert_text, references=texts)
        scores[budget] = score['rouge1']
    
    return scores

total_scores = {}

batch_size = 2
with torch.no_grad():
    # for i in tqdm(range(0, len(val_datasets), batch_size)):
    for i in tqdm(range(0, 200, batch_size)):
        examples = train_datasets[i:i+batch_size]
        scores = m(examples)
        for budget in privacy_budgets:
            if budget not in total_scores:
                total_scores[budget] = []
            total_scores[budget].append(scores[budget])
            
for budget in privacy_budgets:
    print(f"Budget: {budget}")
    print(1-np.mean(total_scores[budget]))
