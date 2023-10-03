import glob
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import os

# Custom Dataset class for MuTual dataset
class MuTualDataset(Dataset):
    def __init__(self, subset, tokenizer, max_length=512):
        # assert subset in ['train', 'test', 'dev']
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.files = os.listdir(os.path.join('..', 'dataset', 'mutual', subset))
        self.subset = subset

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join('..', 'dataset', 'mutual', self.subset, self.files[idx])
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        context = data['article']
        choices = data['options']
        answer_idx = ord(data['answers']) - ord('A')

        input_ids = []
        attention_masks = []
        token_type_ids = []
        for choice in choices:
            text = context + " " + choice
            encoded_dict = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_token_type_ids=True
            )
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])
            token_type_ids.append(encoded_dict['token_type_ids'])

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_masks, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'labels': torch.tensor(answer_idx, dtype=torch.long)
        }

if __name__ == "__main__":
    # Instantiate tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Create datasets
    train_dataset = MuTualDataset(subset='train', tokenizer=tokenizer)
    dev_dataset = MuTualDataset(subset='dev', tokenizer=tokenizer)
    test_dataset = MuTualDataset(subset='test', tokenizer=tokenizer)

    print(len(train_dataset))
    print(len(dev_dataset))
    print(len(test_dataset))

    for x in train_dataset:
        print(x)
        break