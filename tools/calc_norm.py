
from utils.dataset import get_glue_dataset

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import transformers

import torch
import torch.utils.data


model_path = "Llama-3.2-1B-Instruct"
# model_path = "Qwen2-1.5B-Instruct"


model = AutoModelForSequenceClassification.from_pretrained(model_path, device_map='cuda', torch_dtype=torch.bfloat16)
embed_layer = model.get_input_embeddings()
tokenizer = AutoTokenizer.from_pretrained(model_path)
# tokenizer.pad_token = tokenizer.eos_token

dataset, _ = get_glue_dataset("mrpc")
train_dataset = dataset["train"]

train_dataset = train_dataset.map(lambda x: tokenizer(x['sentence1'], x['sentence2'], truncation=True, padding='max_length', max_length=128), batched=True)

dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, collate_fn=transformers.default_data_collator)

norms = []
for batch in dataloader:
    input_ids = batch['input_ids'].cuda()
    embeddings = embed_layer(input_ids)
    norm = torch.norm(embeddings, p=2, dim=-1).max()
    norms.append(norm.item())
    
print(f'norms: {norms}')
print(f'mean norm: {sum(norms) / len(norms)}')
print(f'max norm: {max(norms)}')
print(f'min norm: {min(norms)}')

    