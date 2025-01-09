from itertools import product

class SEQTokenizer_Kmers:
    """Custom tokenizer for DNA sequences with k-mers."""
    def __init__(self, k):
        self.k = k
        #self.max_length = max_length
        #self.pad_token_id = pad_token_id
        
        self.vocab = [''.join(kmer) for kmer in product('ATCG', repeat=k)]
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}

    def encode(self, sequence):
        kmers = [sequence[i:i+self.k] for i in range(len(sequence) - self.k + 1)]
        #token_ids = [self.token_to_id[kmer] for kmer in kmers if kmer in self.token_to_id]

        # Truncate to max_length 
        #token_ids = token_ids[:self.max_length]

        # Pad if the sequence is shorter than max_length
        #if len(token_ids) < self.max_length:
            #token_ids += [self.pad_token_id] * (self.max_length - len(token_ids))
        
        return [self.token_to_id[kmer] for kmer in kmers if kmer in self.token_to_id]
        #return token_ids #instead of the line above 
