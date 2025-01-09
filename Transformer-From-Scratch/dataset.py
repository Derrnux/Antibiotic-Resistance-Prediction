import torch

class SEQDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, labels, tokenizer):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        tokens = self.tokenizer.encode(sequence)
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(label, dtype=torch.long)