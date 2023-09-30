import torch
import numpy as np
import os
import json
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from nltk import wordpunct_tokenize

class MutualDataset(Dataset):

    def __init__(self, dir, tokenizer="bert-base-uncased", root="Data/mutual/", max_length=None):
        self.dirs = dir if type(dir) == list else [dir]
        self.root = root
        self.data = []
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        self.mapping = {"A":0, "B":1, "C":2, "D":3}

        for dir in self.dirs:
            for f in os.listdir(root + dir):
                sample = json.load(open(root + dir + "/" + f))
                self.data.append(sample)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_dict = self.data[idx]

        label = self.mapping[data_dict["answers"]]

        label_tensor = torch.zeros(4)
        label_tensor[label] = 1

        # Combine the answer options and the articles
        sent_0 = data_dict["article"] + data_dict["options"][0]
        sent_1 = data_dict["article"] + data_dict["options"][1]
        sent_2 = data_dict["article"] + data_dict["options"][2]
        sent_3 = data_dict["article"] + data_dict["options"][3]

        # Encode the sentences
        encoded_0 = self.tokenizer.encode_plus(
            text=sent_0,
            add_special_tokens=True,
            max_length=None,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt"
        )

        encoded_1 = self.tokenizer.encode_plus(
            text=sent_1,
            add_special_tokens=True,
            max_length=None,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt"
        )
        encoded_2 = self.tokenizer.encode_plus(
            text = sent_2,
            add_special_tokens=True,
            max_length=None,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt"
        )
        encoded_3 = self.tokenizer.encode_plus(
            text = sent_3,
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
            )
        )

        encoded_0["input_ids"] = sentences
        encoded_0["attention_mask"] = attention_masks
        encoded_0["token_type_ids"] = token_type_ids

        return encoded_0, label_tensor
