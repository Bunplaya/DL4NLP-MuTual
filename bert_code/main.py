import torch
import argparse
from transformers import AutoModelForMultipleChoice, AutoModel, GPT2LMHeadModel
from data import MutualDataset
import os
from torch.utils.data import DataLoader
import numpy as np
import tqdm
import wandb

def load_model(model_name, method="multiple-choice"):
    # TODO: Code for multi-gpu setup not yet working, probably not required, as models are relatively small
    # Load tokenizer
    if model_name == "bert-base-uncased":
        model = AutoModelForMultipleChoice.from_pretrained(model_name)
    elif model_name == "gpt2":
        model = GPT2LMHeadModel.from_pretrained(model_name)
    else:
        raise Exception

    return model
def get_model(download_path, model_version):

    if os.listdir(download_path):
        answer = input("Provided path already contains files, continue anyway [y/n]? \n")
        if answer.lower() == "y":
            model = AutoModel.from_pretrained(model_version)
            model.save_pretrained(download_path)

    else:
        model = AutoModel.from_pretrained(model_version)
        model.save_pretrained(download_path)

    return

def train_model(model, train_dataloader, val_dataloader, max_grad_norm,
                epochs, learning_rate, adam_e, device, freeze=False, use_wandb=False,
                save_dir="Finetuned/bert", results_dir="Results/bert", wandb_name=None):

    if use_wandb:
        wandb.init(
            project="DL4NLP",
            name=wandb_name,
            config={
                "learning_rate":learning_rate,
                "model":save_dir.split("/")[1],
                "epochs":epochs,
                "freeze":freeze
            }
        )

    loss_module = torch.nn.CrossEntropyLoss()

    model.to(device)

    # Freeze backbone of model and only finetune the head
    if freeze:
        for n, p in model.named_parameters():
            if n.split(".")[0] != "classifier":
                p.requires_grad= False

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    training_loss, training_acc = [], []
    val_loss, val_acc = [], []

    print(f"Starting training loop for {epochs} epochs:")

    highest_acc = 0

    best_model = model

    print(epochs)

    for i in range(epochs):
        epoch_loss, epoch_acc = [], []

        for batch, targets in tqdm.tqdm(train_dataloader):
            model.train()

            input_tokens = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)

            targets = targets.to(device)

            preds = model(input_tokens, attention_mask, token_type_ids)[0]

            loss = loss_module(preds, targets)

            epoch_loss.append(loss.item())
            epoch_acc.append(np.mean(torch.argmax(preds, dim=1).cpu().numpy() == torch.argmax(targets, dim=1).cpu().numpy()))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()

            model.zero_grad()


        # Compute training set stats of epoch
        epoch_loss = np.mean(epoch_loss)
        epoch_acc = np.mean(epoch_acc)

        training_loss.append(epoch_loss)
        training_acc.append(epoch_acc)

        # Compute validation set stats
        val_acc_epoch, val_loss_epoch = eval_model(model, val_dataloader, loss_module)
        val_loss.append(val_loss_epoch)
        val_acc.append(val_acc_epoch)

        # Check validation accuracy of model, we want to save model with highest validation acc.
        if val_acc_epoch > highest_acc:
            best_model = model
            highest_acc = val_acc_epoch

        if use_wandb:
            # Log results to wandb
            wandb.log({"val_acc":np.mean(val_acc),
                      "val_loss":np.mean(val_loss),
                      "train_acc":epoch_acc,
                      "train_loss":epoch_loss})

    print(f"Training completed, saving model at:{save_dir}")
    best_model.save_pretrained(save_dir)

    return


def eval_model(model, dataloader, loss_module):
    model.eval()

    acc, loss = [], []

    for batch, targets in dataloader:
        input_tokens = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)

        targets = targets.to(device)

        preds = model(input_tokens, attention_mask, token_type_ids)[0]

        acc.append(np.mean(torch.argmax(preds, dim=1).cpu().numpy() == torch.argmax(targets, dim=1).cpu().numpy()))

        loss_val = loss_module(preds, targets)

        loss.append(loss_val.item())

    return np.mean(acc), np.mean(loss)


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    # Arguments for downloading BERT weights
    argParser.add_argument("--get_model", help="Download weights, only run if weights are not yet downloaded",
                            action="store_true")
    argParser.add_argument("--get_version", help="Which weights to download",
                           type=str)
    argParser.add_argument("--download_path", help="Path to download weights to",
                           type=str)

    argParser.add_argument("--model", help="Model to load from huggingface",
                           type=str, default="bert-base-uncased")
    argParser.add_argument("--model_ckpt", help="Path to model weights",
                           type=str, default="Models/bert-base-uncased")

    # Data Arguments
    argParser.add_argument("--data_tokenizer", help="Which models tokenizer to use for the data",
                           type=str, default="bert-base-uncased")


    # Training Arguments
    argParser.add_argument("--epochs", help="Number of epochs to run training for",
                           type=int, default=10)
    argParser.add_argument("--learning_rate", help="Learning rate to use during training",
                           type=float, default=5e-5)
    argParser.add_argument("--freeze", help="Freeze the backbone and only train classifier head",
                           action="store_true")
    argParser.add_argument("--CPU", help="Use CPU during training",
                           action="store_true")
    argParser.add_argument("--save_dir", help="Where to save the model after training",
                           type=str, default="Finetuned/bert")
    argParser.add_argument("--stats_dir", help="Where to save the training stats",
                           type=str, default="Results/bert")
    argParser.add_argument("--wandb", help="Whether to use weights and biases",
                           action="store_true")
    argParser.add_argument("--train_dir", help="Training data directory",
                            type=str, default="train")
    argParser.add_argument("--val_dir", help="Validation data directory",
                            type=str, default="dev")
    argParser.add_argument("--batch_size", help="Batch size during training",
                           type=int, default=16)
    argParser.add_argument("--adam_e", help="Epsilon for adam optim",
                           type=float, default=1e-8)
    argParser.add_argument("--max_grad_norm", help="",
                           type=float, default=1.0)
    argParser.add_argument("--wandb_name", help="Name for WandB run",
                           type=str, default=None)

    args = argParser.parse_args()


    # Downloads the desired bert version to the machine, only runs if specified
    if args.get_model:
        get_model(args.download_path, args.get_version)

    # if args.model == "bert-base-uncased":
    #     model = args.model
    #     ckpt = "Models/bert-base-uncased"
    # elif args.model == "gpt2":
    #     model = args.model
    #     ckpt = "Models/gpt2"

    if args.CPU:
        print("Training on CPU")
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        print("Training on GPU")
        device = torch.device("cuda")
    else:
        print("GPU not available, training on CPU instead")
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    torch.cuda.empty_cache()

    model = load_model(args.model)

    train_dataset = MutualDataset("train",args.data_tokenizer)
    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True)

    val_dataset = MutualDataset("dev", args.data_tokenizer)
    val_dataloader = DataLoader(val_dataset, args.batch_size, shuffle=True)

    train_model(model, train_dataloader, val_dataloader, args.max_grad_norm,
                args.epochs, args.learning_rate, args.adam_e, device, args.freeze, args.wandb,
                save_dir=args.save_dir, results_dir=args.stats_dir, wandb_name=args.wandb_name)