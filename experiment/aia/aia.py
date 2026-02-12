import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from datasets import DatasetDict
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, root_mean_squared_error
from torch import nn
import torch
import pandas as pd
import os
import argparse
from utils.noise import sample_noise_Chi
from transformers import AutoTokenizer, LlamaForSequenceClassification, Qwen2ForSequenceClassification
from modeling.my.split import SplittedQwen2ForSequenceClassification
from modeling.my.configuration import MyQwen2Config
from modeling.my_llama.split import SplittedLlamaForSequenceClassification
from modeling.my_llama.configuration import MyLlamaConfig
from utils.model import load_classification_model_checkpoint
import datasets
import torch.nn.functional as F

def get_noisy_embedding(inputs, embed_layer, base_batch_size, test_eta):
    input_ids = torch.tensor(inputs['input_ids']).cuda()
    attn_masks = torch.tensor(inputs['attention_mask'])
    # attn_masks = attn_masks.to("cpu")
    init_embeddings = []
    for i in range(0, len(input_ids), base_batch_size):
        this_input_ids = input_ids[i:i+base_batch_size]
        this_init_embeddings = embed_layer(this_input_ids)
        this_init_embeddings = this_init_embeddings.to("cpu")
        init_embeddings.append(this_init_embeddings)
    init_embeddings = torch.cat(init_embeddings)
    print(init_embeddings.shape)
    # sample noise
    if test_eta > 0:
        noises = sample_noise_Chi(init_embeddings.shape, test_eta, "cuda").to("cpu")
        noises = noises.to("cpu")
    else:
        noises = 0
    init_embeddings = init_embeddings + noises
    return init_embeddings, attn_masks



def get_cls_embedding(hid_states, attention_mask):
    last_pad = attention_mask.sum(dim=1) - 1
    last_pad = last_pad.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, hid_states.shape[-1])
    cls_embs = torch.gather(hid_states, index=last_pad, dim=1)
    cls_embs = cls_embs.squeeze()
    return cls_embs


def get_embedding(token_embeddings, attn_masks, model, base_batch_size):
    cls_embs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(token_embeddings), base_batch_size)):
            token_embeds_batch = token_embeddings[i:i+base_batch_size].cuda()
            att_masks_batch = attn_masks[i:i+base_batch_size].cuda()

            outputs = model(
                inputs_embeds=token_embeds_batch, 
                attention_mask=att_masks_batch, 
                output_hidden_states=True)
            
            batch_cls_embs = get_cls_embedding(outputs.hidden_states[-1],
                                               att_masks_batch)
            # batch_cls_embs = get_cls_embedding(outputs.denoise_hidden_states,
            #                                    att_masks_batch)
            batch_cls_embs = batch_cls_embs.cpu()
            cls_embs.append(batch_cls_embs)

    cls_embs = torch.cat(cls_embs, 0)
    return cls_embs


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.embedding(text)
        embedded = embedded.unsqueeze(1)
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)



class AttributeInferenceMLP(nn.Module):
    def __init__(self, input_dim, output_dim=2):  
        super(AttributeInferenceMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 768)  
        self.relu = nn.ReLU()  
        self.fc2 = nn.Linear(768, output_dim)  

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def attribute_inference_attack(train_cls_embs, test_cls_embs, train_labels, test_labels, epoch, batch_size, task_type):
    
    input_dim = train_cls_embs.shape[-1]
    if task_type == "classification":
        n_classes = len(torch.unique(train_labels))
        mlp_model = AttributeInferenceMLP(input_dim, n_classes).cuda()
        loss_fn = nn.CrossEntropyLoss()
    elif task_type == "regression":
        mlp_model = AttributeInferenceMLP(input_dim, 1).cuda()
        loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(mlp_model.parameters(), lr=0.0001)

    epoch_accuracies = []
    epoch_losses = []
    # f1_scores = []

    for epoch in range(epoch):
        mlp_model.train()
        for i in tqdm(range(0, len(train_cls_embs), batch_size)):
            Xbatch = train_cls_embs[i:i+batch_size].cuda()
            ybatch = train_labels[i:i+batch_size].cuda()

            optimizer.zero_grad()
            y_pred = mlp_model(Xbatch).squeeze()
            loss = loss_fn(y_pred, ybatch)
            loss.backward()
            optimizer.step()

        
        mlp_model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for i in range(0, len(test_cls_embs), batch_size):
                Xbatch = test_cls_embs[i:i+batch_size].cuda()
                y_logit = mlp_model(Xbatch)
                if task_type == "classification":
                    y_pred = torch.argmax(y_logit, -1)
                    all_preds.extend(y_pred.cpu().numpy())
                    all_labels.extend(test_labels[i:i+batch_size].float().cpu().numpy())
                elif task_type == "regression":
                    all_preds.extend(y_logit.squeeze().float().cpu().numpy())
                    all_labels.extend(test_labels[i:i+batch_size].float().cpu().numpy())

        if task_type == "classification":
            accuracy = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average='macro')
            tqdm.write(f'Epoch {epoch}: Loss {loss.item()}, Accuracy {accuracy}, F1 {f1}')

            epoch_accuracies.append(accuracy)
            epoch_losses.append(loss.item())
            # f1_scores.append(f1)
        elif task_type == "regression":
            # loss = loss_fn(torch.tensor(all_preds), torch.tensor(all_labels))
            # tqdm.write(f'Epoch {epoch}: Loss {loss.item()}')
            # epoch_losses.append(loss.item())
            loss = root_mean_squared_error(all_labels, all_preds)
            tqdm.write(f'Epoch {epoch}: Loss {loss}')
            epoch_losses.append(loss)

    if task_type == "classification":
        
        # print("Accuracies:", epoch_accuracies)
        # print("f1 scores:", f1_scores)
        # print("Losses:", epoch_losses)
        print("max accuracy:", max(epoch_accuracies))
        # print("max f1:", max(f1_scores))
        return max(epoch_accuracies)
    elif task_type == "regression":
        min_loss = min(epoch_losses)
        print("min loss:", min_loss)
        return min_loss
    else:
        raise ValueError("Invalid task type")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str)
    parser.add_argument("--base_model_ckpt", type=str, default=None)
    # parser.add_argument("--attack_data", type=str)
    parser.add_argument("--test_eta", type=float)
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--attribute", type=str)
    parser.add_argument("--base_batch_size", type=int, default=6)
    parser.add_argument("--write_to_file", type=str)

    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    if 'Qwen' in args.base_model:
        model_cls = SplittedQwen2ForSequenceClassification
        config_cls = MyQwen2Config
    elif 'Llama' in args.base_model:
        model_cls = SplittedLlamaForSequenceClassification
        config_cls = MyLlamaConfig

    if args.base_model_ckpt is None:
        if 'Qwen' in args.base_model:
            model_cls = Qwen2ForSequenceClassification
        elif 'Llama' in args.base_model:
            model_cls = LlamaForSequenceClassification
        base_model = model_cls.from_pretrained(args.base_model, num_labels=2,  device_map="cuda",
                                               pad_token_id=tokenizer.pad_token_id,
                                               ).eval()
    else:
        base_model = load_classification_model_checkpoint(args.base_model_ckpt, 
                                                        tokenizer=tokenizer,
                                                        num_labels=2,
                                                        model_cls=model_cls,
                                                        model_path=args.base_model,
                                                        config_cls=config_cls,
                                                        ).eval().cuda()
                                                        
    
    num_labels = 6
    
    ds = datasets.load_dataset('soda-lmu/tweet-annotation-sensitivity-2')
    
    ds = ds.filter(lambda x: all(v is not None for v in x.values()))
    
    ds = ds.class_encode_column("offensive_language")
    ds = ds['train'].train_test_split(train_size=2000, seed=123, stratify_by_column="offensive_language")['test']
    
    assert args.attribute in ['education', 'age']
    ds = ds.train_test_split(train_size=1600, test_size=400, seed=123, stratify_by_column="offensive_language")
    if args.attribute == 'education':
        # ds = ds.class_encode_column("education")
        task_type = "classification"
    elif args.attribute == 'age':
        task_type = "regression"
    
    # ds = ds.select_columns(["tweet_hashed", args.attribute, "age", "education"])
    ds = ds.select_columns(["tweet_hashed", "age", "education"])
    ds = ds.rename_column("tweet_hashed", "text")
    # ds = ds.rename_column(args.attribute, "label")
    
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
    tokenized_datasets = ds.map(preprocess_function, batched=True)
    if args.attribute == 'education':
        tokenized_datasets = tokenized_datasets.class_encode_column("education")
    
    tokenized_train_datasets = tokenized_datasets["train"]
    tokenized_val_datasets = tokenized_datasets["test"]
    # tokenized_train_datasets = train_datasets.map(preprocess_function, batched=True)
    # tokenized_val_datasets = val_datasets.map(preprocess_function, batched=True)
        
    train_inputs,train_labels = tokenized_train_datasets,tokenized_train_datasets[args.attribute]
    test_inputs,test_labels = tokenized_val_datasets,tokenized_val_datasets[args.attribute]
    train_labels = torch.tensor(train_labels).cuda()
    test_labels = torch.tensor(test_labels).cuda()

    scores = []

    for _ in range(5):
        # obtain the embeddings for clean embedding
        #train_cls_embs = get_embedding(train_inputs, base_model)
        #test_cls_embs = get_embedding(test_inputs, base_model)

        #inital attribute inference attack before privatization
        #print("original attribute inference accuracy:")
        #attribute_inference_attack(train_cls_embs, test_cls_embs)

        
    
        with torch.no_grad():
            train_token_embeddings, train_attn_masks = get_noisy_embedding(train_inputs, base_model.get_input_embeddings(),
                                                                        args.base_batch_size, args.test_eta)
            test_token_embeddings, test_attn_masks = get_noisy_embedding(test_inputs, base_model.get_input_embeddings(),
                                                                        args.base_batch_size, args.test_eta)

            train_cls_embs = get_embedding(train_token_embeddings, train_attn_masks, base_model, args.base_batch_size)
            test_cls_embs = get_embedding(test_token_embeddings, test_attn_masks, base_model, args.base_batch_size)

        print(f"\nTraining with differential privacy budget = {args.test_eta}")
        
        for _ in range(5):
            acc = attribute_inference_attack(train_cls_embs, test_cls_embs, train_labels, test_labels, args.epoch, args.batch_size, task_type)
            scores.append(acc)
        
    acc = sum(scores) / len(scores)
    with open(args.write_to_file, 'a') as f:
        f.write(f"{args.test_eta},{acc}\n")