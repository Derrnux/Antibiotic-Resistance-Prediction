from itertools import product

class SEQTokenizer_Kmers:
    """Custom tokenizer for DNA sequences with k-mers."""
    def __init__(self, k):
        self.k = k
        
        self.vocab = [''.join(kmer) for kmer in product('ATCG', repeat=k)]
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}

    def encode(self, sequence):
        kmers = [sequence[i:i+self.k] for i in range(len(sequence) - self.k + 1)]
        return [self.token_to_id[kmer] for kmer in kmers if kmer in self.token_to_id]