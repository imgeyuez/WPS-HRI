import torch
from torch.utils.data import Dataset, DataLoader

# Load the processed dataset
traindataset = torch.load(r"C:\Users\imgey\Desktop\MASTER_POTSDAM\WiSe2425\PM1_argument_mining\train_dataset.pt")
devdataset = torch.load(r"C:\Users\imgey\Desktop\MASTER_POTSDAM\WiSe2425\PM1_argument_mining\dev_dataset.pt")

class NERDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.data[idx]["input_ids"]),
            "attention_mask": torch.tensor(self.data[idx]["attention_mask"]),
            "labels": torch.tensor(self.data[idx]["labels"])
        }

# Create DataLoader
batch_size = 16
train_dataset = NERDataset(traindataset)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_dataset = NERDataset(devdataset)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True)
