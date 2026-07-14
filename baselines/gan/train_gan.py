import os
from pathlib import Path

from tqdm import tqdm
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import datasets
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Gamma
from torch.autograd import Variable
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModelForSeq2SeqLM

import matplotlib.pyplot as plt

from utils.noise import get_noisy_embedding
from utils.dataset import get_glue_dataset
from utils.model import setup_seed
from baselines.gan.gan_modeling import Generator, Discriminator


setup_seed(12399)


writer: SummaryWriter = None


def load_first_available_dataset(candidates):
    errors = []
    for candidate in candidates:
        try:
            if isinstance(candidate, tuple):
                name, config = candidate
                return datasets.load_dataset(name, config)
            return datasets.load_dataset(candidate)
        except Exception as exc:
            errors.append(f"{candidate}: {type(exc).__name__}: {str(exc).splitlines()[0][:160]}")
    raise RuntimeError("Unable to load dataset from candidates:\n" + "\n".join(errors))


def pretrain_generator(g_model, data_loader, embed_layer, privacy_budget, 
                       epochs=200, lr=0.0002, device="cuda", llm_type="qwen2-1.5b"):
    g_model = g_model.to(device)
    
    
    criterion = nn.MSELoss()
    g_optimizer = optim.AdamW(g_model.parameters(), lr=lr)

    losses = []

    
    for epoch in tqdm(range(epochs)):
        epoch_loss = 0
        for i, batch in enumerate(data_loader):
            input_ids = batch['input_ids'].to(device)
            
            attention_mask = None
            
            real_embeddings = embed_layer(input_ids)
            noisy_embeddings = get_noisy_embedding(real_embeddings, privacy_budget, clip=True, model_type=llm_type)[0]
            
            batch_size = real_embeddings.size(0)
            seq_len = real_embeddings.size(1)
            hidden_size = real_embeddings.size(2)


            
            real_imgs = Variable(real_embeddings.view(-1, hidden_size))

            
            
            

            g_optimizer.zero_grad()

            
            z = Variable(noisy_embeddings)

            
            gen_imgs = g_model(z, attention_mask)
            gen_imgs = gen_imgs.view(-1, hidden_size)

            
            g_loss = criterion(gen_imgs, real_imgs)
            g_loss.backward()
            g_optimizer.step()

            epoch_loss += g_loss.item()
            
            writer.add_scalar("Pretain G loss", g_loss.item(), i+epoch*len(data_loader))
        
        epoch_loss /= len(data_loader)
        losses.append(epoch_loss)
        
        writer.flush()
        
        tqdm.write(f"EPOCH {epoch}: G loss: {epoch_loss}")

    return g_model


def train_gan(g_model, d_model, data_loader, 
              privacy_budget, embed_layer, epochs=200, 
              d_lr=0.0002, g_lr=0.0002, save_path=None, 
              device="cuda", save_every=5, llm_type="qwen2-1.5b"):
    g_model = g_model.to(device)
    d_model = d_model.to(device)
    
    
    criterion = nn.BCELoss()
    d_optimizer = optim.AdamW(d_model.parameters(), lr=d_lr)
    g_optimizer = optim.AdamW(g_model.parameters(), lr=g_lr)

    d_losses = []
    g_losses = []

    
    for epoch in tqdm(range(epochs)):
        epoch_g_loss = 0
        epoch_d_loss = 0
        for i, batch in enumerate(data_loader):  
            input_ids = batch['input_ids'].to(device)
            
            attention_mask = None
            
            real_embeddings = embed_layer(input_ids)
            noisy_embeddings = get_noisy_embedding(real_embeddings, privacy_budget, clip=True, model_type=llm_type)[0]
            
            batch_size = real_embeddings.size(0)
            seq_len = real_embeddings.size(1)
            hidden_size = real_embeddings.size(2)
            

            
            valid = Variable(torch.Tensor(batch_size*seq_len, 1).fill_(1.0).to(device), requires_grad=False)
            fake = Variable(torch.Tensor(batch_size*seq_len, 1,).fill_(0.0).to(device), requires_grad=False)

            
            real_imgs = Variable(real_embeddings.view(-1, hidden_size))

            
            
            

            g_optimizer.zero_grad()

            
            z = Variable(noisy_embeddings)

            
            gen_imgs = g_model(z, attention_mask)
            gen_imgs = gen_imgs.view(-1, hidden_size)

            
            g_loss = criterion(d_model(gen_imgs), valid)
            g_loss2 = nn.MSELoss()(gen_imgs, real_imgs)
            g_total_loss = g_loss + g_loss2
            g_total_loss.backward()
            g_optimizer.step()

            
            
            

            d_optimizer.zero_grad()

            
            real_loss = criterion(d_model(real_imgs), valid)
            fake_loss = criterion(d_model(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            d_optimizer.step()


            epoch_g_loss += g_total_loss.item()
            epoch_d_loss += d_loss.item()
            
            writer.add_scalar("G loss", g_total_loss.item(), i+epoch*len(data_loader))
            writer.add_scalar("D loss", d_loss.item(), i+epoch*len(data_loader))
            
        epoch_g_loss /= len(data_loader)
        epoch_d_loss /= len(data_loader)
        if epoch_g_loss > 90:
            print("G loss too large")
            break
        
        d_losses.append(epoch_d_loss)
        g_losses.append(epoch_g_loss)
        
        writer.flush()
        
        tqdm.write(f"EPOCH {epoch}: G loss: {epoch_g_loss}, D loss: {epoch_d_loss}")
        if save_path is not None and epoch % save_every == 0:
            path = f"{save_path}/epoch_{epoch}"
            os.makedirs(path, exist_ok=True)
            torch.save(g_model.state_dict(), f"{path}/generator.pth")
            torch.save(d_model.state_dict(), f"{path}/discriminator.pth")
            
            plt.plot(d_losses, label="D loss")
            plt.plot(g_losses, label="G loss")
            plt.legend()
            plt.savefig(f"{path}/losses.png")
            plt.close()

    return g_model, d_model


import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_path", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--privacy_budget", type=int, default=6000)
    parser.add_argument("--pretrain_epochs", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=2)
    parser.add_argument("--train_epochs", type=int, default=20)
    parser.add_argument(
        "--financial_phrasebank_config",
        type=str,
        default="sentences_50agree",
        choices=["sentences_50agree", "sentences_allagree"],
    )
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    
    llm_path = args.llm_path
    dataset_name = args.dataset_name
    batch_size = args.batch_size
    privacy_budget = args.privacy_budget
    pretrain_epochs = args.pretrain_epochs
    save_every = args.save_every
    train_epochs = args.train_epochs
    
    
    if "Qwen" in llm_path:
        llm_type = "qwen2-1.5b"
        save_path = (
            Path(args.output_dir)
            if args.output_dir is not None
            else Path(__file__).parent / f"gan_models/{dataset_name}_privacy_{privacy_budget}"
        )
    elif "Llama" in llm_path:
        llm_type = "llama-3.2-1b"
        save_path = (
            Path(args.output_dir)
            if args.output_dir is not None
            else Path(__file__).parent / f"gan_models_llama/{dataset_name}_privacy_{privacy_budget}"
        )
    elif "t5" in llm_path:
        llm_type = "t5-large"
        save_path = (
            Path(args.output_dir)
            if args.output_dir is not None
            else Path(__file__).parent / f"gan_models_t5/{dataset_name}_privacy_{privacy_budget}"
        )
    else:
        raise ValueError("Unknown LLM model")
    
    global writer
    writer = SummaryWriter(log_dir=save_path/"logs")
    
    config = AutoConfig.from_pretrained(llm_path)
    tokenizer = AutoTokenizer.from_pretrained(llm_path)
    if "t5" not in llm_path and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if "t5" in llm_path:
        llm = AutoModelForSeq2SeqLM.from_pretrained(llm_path, config=config)
        llm_hidden_size = llm.config.d_model
    else:
        llm = AutoModelForCausalLM.from_pretrained(llm_path, config=config)
        llm_hidden_size = llm.config.hidden_size
    embed_layer = llm.get_input_embeddings()
    embed_layer = nn.Embedding.from_pretrained(embed_layer.weight).cuda()
    del llm
    

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
            args.financial_phrasebank_config,
            revision="main",
        )["train"]
        if args.financial_phrasebank_config == "sentences_allagree":
            expected_size = 1811 + 226 + 227
            if len(ds) != expected_size:
                raise ValueError(
                    f"Expected financial_phrasebank/{args.financial_phrasebank_config} "
                    f"to contain {expected_size} rows, got {len(ds)}"
                )
            ds = ds.train_test_split(
                train_size=1811,
                test_size=226 + 227,
                seed=123,
                stratify_by_column="label",
            )
            train_datasets = ds["train"]
        else:
            ds = ds.train_test_split(test_size=0.1, seed=123, stratify_by_column="label")
            train_datasets = ds["train"]
        max_length = args.max_length if args.max_length is not None else 256
        def preprocess_function(examples):
            return tokenizer(
                examples["sentence"],
                truncation=True,
                padding="max_length",
                return_tensors="pt",
                max_length=max_length,
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
        if dataset_name == "dailymail":
            local_path = Path("dataset/cnn_dailymail_short")
            ds = (
                datasets.load_dataset(local_path.as_posix())
                if local_path.exists()
                else load_first_available_dataset(
                    [
                        "cnn_dailymail_short",
                        "determined-ai/cnn_dailymail_short",
                    ]
                )
            )

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
            return tokenizer(
                [prefix + inp for inp in examples[input_field]],
                truncation=True,
                padding="max_length",
                return_tensors="pt",
                max_length=input_field_len,
            )
    
    tokenized_datasets = train_datasets.map(preprocess_function, batched=True)
    data_loader = torch.utils.data.DataLoader(tokenized_datasets, batch_size=batch_size, shuffle=True, collate_fn=transformers.default_data_collator)
    

    G = Generator(embedding_size=llm_hidden_size, hidden_size=llm_hidden_size, num_layers=2)

    D = Discriminator(input_size=llm_hidden_size)


    G = pretrain_generator(G, data_loader, embed_layer, 
                           privacy_budget=privacy_budget, epochs=pretrain_epochs, lr=0.0001,
                            llm_type=llm_type,)


    G, D = train_gan(G, D, data_loader, privacy_budget=privacy_budget, embed_layer=embed_layer, epochs=train_epochs,
                     d_lr=0.00005, g_lr=0.0002, 
                     save_path=save_path, save_every=save_every,
                        llm_type=llm_type,
                     )


    writer.close()

if __name__ == "__main__":
    main()
