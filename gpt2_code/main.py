import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from data import MutualDataset
import os
from torch.utils.data import DataLoader
import numpy as np
import tqdm
import wandb

def load_model(cfg_ckpt, method="causal-lm"):
    if method == "causal-lm":
        model = AutoModelForCausalLM.from_pretrained(cfg_ckpt)
    else:
        raise Exception("Provide a valid Method for the model i.e.: causal-lm")

    return model

def train_model(model, train_dataloader, val_dataloader,
                epochs, learning_rate, device, use_wandb=False,
                save_dir="Finetuned/gpt2", results_dir="Results/gpt2"):

    if use_wandb:
        wandb.init(
            project="DL4HLP",
            config={
                "learning_rate": learning_rate,
                "model": save_dir.split("/")[1],
                "epochs": epochs
            }
        )

    loss_module = torch.nn.CrossEntropyLoss()

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    training_loss, val_loss = [], []

    print(f"Starting training loop for {epochs} epochs:")

    highest_val_loss = float("inf")

    best_model = model

    for i in range(int(epochs)):
        epoch_loss = []

        for batch in tqdm.tqdm(train_dataloader):
            model.train()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            targets = input_ids.clone()

            optimizer.zero_grad()

            print(f"Input IDs shape: {input_ids.shape}")
            print(f"Attention Mask shape: {attention_mask.shape}")

            logits = model(input_ids, attention_mask=attention_mask).logits

            loss = loss_module(logits.view(-1, logits.shape[-1]), targets.view(-1))

            epoch_loss.append(loss.item())

            loss.backward()

            optimizer.step()

        epoch_loss = np.mean(epoch_loss)
        training_loss.append(epoch_loss)

        val_loss_epoch = eval_model(model, val_dataloader, loss_module, device)
        val_loss.append(val_loss_epoch)

        if val_loss_epoch < highest_val_loss:
            best_model = model
            highest_val_loss = val_loss_epoch

        if use_wandb:
            wandb.log({"val_loss": val_loss_epoch, "train_loss": epoch_loss})

    print(f"Training completed, saving model at:{save_dir}")
    best_model.save_pretrained(save_dir)

    return




def eval_model(model, dataloader, loss_module):
    model.eval()

    loss = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        targets = input_ids.clone()

        logits = model(input_ids, attention_mask=attention_mask).logits

        loss_val = loss_module(logits.view(-1, logits.shape[-1]), targets.view(-1))

        loss.append(loss_val.item())

    return np.mean(loss)

def collate_batch(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()

    # Training Arguments
    argParser.add_argument("--model", help="Model to load from huggingface",
                       type=str, default="gpt2")
    argParser.add_argument("--epochs", help="Number of epochs to run training for")
    argParser.add_argument("--learning_rate", help="Learning rate to use during training",
                           type=float, default=1e-3)
    argParser.add_argument("--CPU", help="Use CPU during training",
                           action="store_true")
    argParser.add_argument("--save_dir", help="Where to save the model after training",
                           type=str, default="Finetuned/gpt2")
    argParser.add_argument("--stats_dir", help="Where to save the training stats",
                           type=str, default="Results/gpt2")
    argParser.add_argument("--wandb", help="Whether to use weights and biases",
                           action="store_true")
    argParser.add_argument("--freeze", help="Freeze the backbone and only train classifier head",
                       action="store_true")


    # Data Arguments
    argParser.add_argument("--train_dir", help="Training data directory",
                            type=str, default="train")
    argParser.add_argument("--val_dir", help="Validation data directory",
                            type=str, default="dev")
    argParser.add_argument("--batch_size", help="Batch size during training",
                           type=int, default=16)

    args = argParser.parse_args()

    if args.CPU:
        print("Training on CPU")
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        print("Training on GPU")
        device = torch.device("cuda")
    else:
        print("GPU not available, training on CPU instead")
        device = torch.device("cpu")

    # Create the tokenizer and model instances
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Add a special token for padding
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    train_dataset = MutualDataset(args.train_dir, tokenizer, args.model)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)

    val_dataset = MutualDataset(args.val_dir, tokenizer, args.model)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)

    train_model(model, train_dataloader, val_dataloader,
                args.epochs, args.learning_rate, device, args.freeze, args.wandb)

