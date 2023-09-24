import os
import json

class MuTualDataset:
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.data = self.load_data()

    def load_data(self):
        data = []
        for filename in os.listdir(self.data_folder):
            if filename.endswith(".txt"):
                with open(os.path.join(self.data_folder, filename), "r") as f:
                    dialog_data = json.load(f)
                    data.append(dialog_data)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Example usage:
# -----------------------------------------------------------------------------
data_folder = "/path/to/mutual_dataset/train"
dataset = MuTualDataset(data_folder)

for idx in range(len(dataset)):
    dialogue = dataset[idx]
    answers = dialogue["answers"]
    options = dialogue["options"]
    article = dialogue["article"]
    dialog_id = dialogue["id"]

# -----------------------------------------------------------------------------

