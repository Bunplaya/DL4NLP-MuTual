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

        sent_0 = data_dict["article"] + data_dict["options"][0]
        sent_1 = data_dict["article"] + data_dict["options"][1]
        sent_2 = data_dict["article"] + data_dict["options"][2]
        sent_3 = data_dict["article"] + data_dict["options"][3]

        encoded_0 = self.tokenizer.encode_plus(
            text=sent_0,
            add_special_tokens=True,
            max_length=None,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt"
        )

        sentences = torch.cat(
            (
                encoded_0["input_ids"],
                encoded_1["input_ids"],
                encoded_2["input_ids"],
                encoded_3["input_ids"]
            ), dim=0
        )

        attention_masks = torch.cat(
            (
                encoded_0["attention_mask"],
                encoded_1["attention_mask"],
                encoded_2["attention_mask"],
                encoded_3["attention_mask"]
            ), dim=0
        )

        token_type_ids = torch.cat(
            (
                encoded_0["token_type_ids"],
                encoded_1["token_type_ids"],
                encoded_2["token_type_ids"],
                encoded_3["token_type_ids"]
            ), dim=0
        )

        encoded_0["input_ids"] = sentences
        encoded_0["attention_mask"] = attention_masks
        encoded_0["token_type_ids"] = token_type_ids

        return encoded_0, label_tensor
