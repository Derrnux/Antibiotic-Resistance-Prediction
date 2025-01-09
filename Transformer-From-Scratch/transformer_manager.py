from load_data import SEQData
from dataset import SEQDataset
from class_balancing import Balancing
from tokenizer import SEQTokenizer_Kmers
from classifier import SEQClassifier

from collections import Counter
from sklearn.metrics import classification_report
import torch
import random

class Transformer:
    def __init__(self, k_mers_type, attention_layers, batch_size, balancing_method, embed_dim, dim_feedforward, seed):
        self.tokenizer = SEQTokenizer_Kmers(k_mers_type)
        self.class_weights = torch.tensor([1.0, 1.0])
        self.attention_layers = attention_layers
        self.batch_size = batch_size
        self.embed_dim = embed_dim
        self.dim_feedforward = dim_feedforward
        self.balancing_method = balancing_method
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.seed = seed

        self.set_seed(self.seed)
        self.prepare_data()
        self.model = SEQClassifier(self.vocab_size, embed_dim = self.embed_dim, attention_layers = self.attention_layers, dim_feedforward = self.dim_feedforward, seed = self.seed)
        self.model.to(self.device)

    def set_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)
        if self.device == 'cuda':
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def prepare_data(self):
        data = SEQData(self.seed)

        self.seq_train = data.train_sequences
        self.y_train = data.train_labels
        self.seq_val = data.val_sequences
        self.y_val = data.val_labels
        self.seq_test = data.test_sequences
        self.y_test = data.test_labels

        if (self.balancing_method == 'oversampling'):
            self.seq_train, self.y_train = Balancing(self.seed).oversample_data(self.seq_train, self.y_train)
        elif (self.balancing_method == 'undersampling'):
            self.seq_train, self.y_train = Balancing(self.seed).undersample_data(self.seq_train, self.y_train)
        elif (self.balancing_method == 'class_weights'):
            class_counts = Counter(self.y_train)
            self.class_weights = torch.tensor([1.0 / class_counts[0], 1.0 / class_counts[1]])

        self.train_dataset = SEQDataset(self.seq_train, self.y_train, self.tokenizer)
        self.val_dataset = SEQDataset(self.seq_val, self.y_val, self.tokenizer)
        self.test_dataset = SEQDataset(self.seq_test, self.y_test, self.tokenizer)

        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle = True)
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size = self.batch_size, shuffle = True)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size = self.batch_size, shuffle = True)

        self.vocab_size = len(self.tokenizer.vocab)

    def train(self, epochs, learning_rate):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        loss_fn = torch.nn.CrossEntropyLoss(self.class_weights.to(self.model.fc.weight.device))

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for sequences, labels in self.train_loader:
                sequences, labels = sequences.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(sequences)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    def evaluate(self):
        self.model.eval()
        
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for sequences, labels in self.val_loader:
                sequences, labels = sequences.to(self.device), labels.to(self.device)
                outputs = self.model(sequences)
                preds = torch.argmax(torch.nn.functional.softmax(outputs, dim=1), dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        report = classification_report(all_labels, all_preds, zero_division=0, output_dict=True) 
        print(report)
        return report['macro avg']['f1-score']

    def test(self):
        self.model.eval()
        
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for sequences, labels in self.test_loader:
                sequences, labels = sequences.to(self.device), labels.to(self.device)
                outputs = self.model(sequences)
                preds = torch.argmax(torch.nn.functional.softmax(outputs, dim=1), dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        print(f'Labels:      {all_labels}')
        print(f'Predictions: {all_preds}')

        print(classification_report(all_labels, all_preds, zero_division = 0))