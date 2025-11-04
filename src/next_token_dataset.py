import torch
import pandas as pd
import os
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from torch.utils.data import random_split

BASE_DIR = Path(__file__).resolve().parent.parent
DEBUG = True

def create_train_val_test_splits(tokenized_tweets, seq_length=140, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """Create train, validation and test splits with consistent vocabulary

    Args:
        tokenized_tweets (Pandas Series): full dataset of tweets
        seq_length (int, optional): limit for length of tweet, default=140.
        train_ratio (float, optional): part for train. Defaults to 0.8.
        val_ratio (float, optional): part for validation. Defaults to 0.1.
        test_ratio (float, optional): part for test. Defaults to 0.1.
        seed (int, optional): random seed val. Defaults to 42.

    Returns:
        NextTokenDataset, NextTokenDataset, NextTokenDataset, dict: train, val, test datasets and full vocab
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    
    # trim data for debug mode
    if DEBUG:
        tokenized_tweets = tokenized_tweets.sample(n=100)

    # Create full dataset first to get consistent vocabulary
    full_dataset = NextTokenDataset(tokenized_tweets, seq_length=seq_length)
    vocab = full_dataset.get_vocab()
    
    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    return train_dataset, val_dataset, test_dataset, vocab

class NextTokenDataset(Dataset):
    def __init__(self, tokenized_tweets, seq_length=140, pad_to_seq_length=True):
        """
        Args:
            tokenized_tweets: List of tokenized tweets
            seq_length: Length of input sequence
        """
        self.tokenized_tweets = tokenized_tweets
        self.seq_length = seq_length
        self.pad_to_seq_length = pad_to_seq_length
        
        # Build vocabulary
        self.vocab = self._build_vocab()
        
        # Create sequences
        self.sequences, self.targets, self.masks = self._create_sequences()
    
    def _build_vocab(self):
        """Build vocabulary from all tokens"""
        counter = Counter()
        for tweet in self.tokenized_tweets:
            counter.update(tweet)
        
        vocab = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<SOS>': 2,
            '<EOS>': 3
        }
        
        for word, count in counter.items():
            if word not in vocab:
                vocab[word] = len(vocab)
        
        return vocab
    
    def _create_sequences(self):
        """Create input sequences and target tokens"""
        sequences = []
        targets = []
        masks = []
        
        for tweet in self.tokenized_tweets:
            if len(tweet) <= 1:
                continue
                
            # Convert tokens to indices
            tweet_indices = [self.vocab.get(token, self.vocab['<UNK>']) for token in tweet]
            

            # Create sliding window sequences
            len_seq = min(len(tweet_indices), self.seq_length)
            for i in range(0, len_seq):
                seq = tweet_indices[0:i]
                target = tweet_indices[i:i + 1]
                
                targets.append(target)

                if self.pad_to_seq_length:
                    # Pad sequence to seq_length
                    padded_seq = seq + [self.vocab['<PAD>']] * (self.seq_length - len(seq))
                    
                    # Create attention mask (1 for real tokens, 0 for padding)
                    mask = [1] * len(seq) + [0] * (self.seq_length - len(seq))

                    sequences.append(padded_seq)
                    masks.append(mask)
                else:
                    sequences.append(seq)
                    
        
        return sequences, targets, masks
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.sequences[idx], dtype=torch.long),
            'target': torch.tensor(self.targets[idx], dtype=torch.long),
            'masks': torch.tensor(self.masks[idx], dtype=torch.long)
        }
    
    def get_vocab_size(self):
        return len(self.vocab)
    
    def get_vocab(self):
        return self.vocab
    
if __name__ == '__main__':
    # test run
    
    df_tweets = pd.read_csv(os.path.join(BASE_DIR, 'data', 'tokens_tweets.csv'))
    # for test aim - only 100 samples
    df_tweets = df_tweets.sample(n=100).reset_index(drop=True)
    
    test_dataset = NextTokenDataset(df_tweets['tokens'])
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)
    first_batch = next(iter(test_loader))
    print(f'print first batch, batch_size = {first_batch["input_ids"].shape}')
    print(first_batch)