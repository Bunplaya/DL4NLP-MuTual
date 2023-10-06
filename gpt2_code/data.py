import torch
import os
import json
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class MutualDataset(Dataset):

    def __init__(self, data_dir, tokenizer, model_version, max_length=128):
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

        label_tensor = torch.zeros(4)
        label_tensor[label] = 1

        input_text = data_dict["article"] + " " + " ".join(data_dict["options"])

        encoded = self.tokenizer.encode_plus(
            text=input_text,
            add_special_tokens=True,
            max_length=None,  # Remove max_length constraint
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt"
        )

        # Get the maximum sequence length in the batch
        max_length = encoded["input_ids"].shape[1]

        # Adjust the max_length for padding
        self.max_length = max_length

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "label": label_tensor
        }
