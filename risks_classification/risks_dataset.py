from torch.utils.data import Dataset
import torch


class RisksDataset(Dataset):
    # To use this class data_path should be the path to the TSV file in format "label \t description"
    def __init__(self, data_path, tokenizer):
        self.labels_map = {
            'fall': 0,
            'burn': 1,
            'energy_wasting': 2,
            'water_wasting': 3,
            'none_risk': 4
        }
        self.descriptions = []
        self.labels = []
        self.tokenizer = tokenizer
        # with open(data_path, 'r') as f:
        #     for line in f:
        #         data = line.strip().split('\t')
        #         self.descriptions.append(data[1])
        #         self.labels.append(self.labels_map[data[0]])
        self.descriptions = data_path['description'].values
        self.labels = data_path['risk'].apply(lambda x: self.labels_map[x]).values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        description = self.descriptions[idx]
        label = self.labels[idx]

        inputs = self.tokenizer(description, max_length=128, truncation=True, padding="max_length", return_tensors="pt")
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label)
        }