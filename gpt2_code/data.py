import torch
import os
import json
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class MutualDataset(Dataset):

    def __init__(self, data_dir, tokenizer, model_version, max_length=1024):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.model_version = model_version
        self.max_length = max_length
        self.mapping = {"A": 0, "B": 1, "C": 2, "D": 3}

        self.data = []

        for filename in os.listdir(self.data_dir):
            with open(os.path.join(self.data_dir, filename), 'r') as file:
                sample = json.load(file)
                self.data.append(sample)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_dict = self.data[idx]

        label = self.mapping[data_dict["answers"]]

        label_tensor = torch.tensor(label)

        input_text = data_dict["article"] + " " + " ".join(data_dict["options"])

        # Tokenize and pad the input text
        encoded = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze(),
            "label": label_tensor
        }