import torch

class SEQClassifier(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, attention_layers, dim_feedforward, seed):
        super(SEQClassifier, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim)
        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, dim_feedforward=dim_feedforward, activation='gelu'),
            num_layers = attention_layers
        )
        self.fc = torch.nn.Linear(embed_dim, 2)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x