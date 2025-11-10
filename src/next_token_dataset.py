import torch
import pandas as pd
import os
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast


BASE_DIR = Path(__file__).resolve().parent.parent
PART_TARGET = 0.25
DEBUG = False


class NextTokenDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_length=140):
        """
        Args:
            texts: input lisf of text
            tokenizer: List of tokenized tweets
            seq_length: Length of input sequence
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        
        # Create sequences
        self.sequences, self.targets, self.masks = self._create_sequences()
    
    def _create_sequences(self):
        """Create input sequences and target tokens"""
        sequences = []
        targets = []
        masks = []
        
        for tweet in self.texts:
            if len(tweet) <= 1:
                continue

            encoded = self.tokenizer(
                tweet,
                max_length=self.seq_length,
                truncation=True
            )
            
            input_ids = encoded['input_ids']

            # Create sliding window sequences
            len_seq = min(len(input_ids), self.seq_length)
            for i in range(1, len_seq-1):
                
                seq = input_ids[:i]
                target = input_ids[i]
                mask = [1] * len(seq)
                
                # Add padding
                padded_seq = seq + [0] * (self.seq_length - len(seq))
                padded_mask = mask + [0] * (self.seq_length - len(mask))
                
                sequences.append(padded_seq)
                targets.append(target)
                masks.append(padded_mask)
                    
        
        return sequences, targets, masks
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.sequences[idx], dtype=torch.long),
            'target': torch.tensor(self.targets[idx], dtype=torch.long),
            'masks': torch.tensor(self.masks[idx], dtype=torch.long)
        }

class ValTokenDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_length=140):
        """
        Args:
            texts: input lisf of text
            tokenizer: List of tokenized tweets
            seq_length: Length of input sequence
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        
        # Create sequences
        self.sequences, self.targets, self.masks = self._create_sequences()
    
    def _create_sequences(self):
        """Create input sequences and target tokens"""
        sequences = []
        targets = []
        masks = []
        
        for tweet in self.texts:
            if len(tweet) <= 1:
                continue

            encoded = self.tokenizer(
                tweet,
                max_length=self.seq_length,
                truncation=True
            )
            
            input_ids = encoded['input_ids']
            n_target = int(len(input_ids) * PART_TARGET)
            if n_target < 1:
                n_target = 1
            
            seq = input_ids[:-n_target]
            target = input_ids[-n_target:]
            mask = [1] * len(seq)

            # Add padding
            padded_seq = seq + [0] * (self.seq_length - len(seq))
            padded_target = target + [0] * (self.seq_length - len(target))
            padded_mask = mask + [0] * (self.seq_length - len(mask))
                
            sequences.append(padded_seq)
            targets.append(padded_target)
            masks.append(padded_mask)
        
        return sequences, targets, masks
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.sequences[idx], dtype=torch.long),
            'target': torch.tensor(self.targets[idx], dtype=torch.long),
            'masks': torch.tensor(self.masks[idx], dtype=torch.long)
        }
    
    
if __name__ == '__main__':
    pass
    # test run
    # df_tweets = pd.read_csv(os.path.join(BASE_DIR, 'data', 'cleaned_tweets.csv'))
    # # for test aim - only 100 samples
    # df_tweets = df_tweets.sample(n=100).reset_index(drop=True)
    
    # tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    # test_dataset = NextTokenDataset(df_tweets['text'], tokenizer, seq_length=140)
    # test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)
    # first_batch = next(iter(test_loader))
    # print(f'print first batch, batch_size = {first_batch["input_ids"].shape}')
    # print(first_batch)
    # print('Test ValTokenDataset')
    # test_dataset = ValTokenDataset(df_tweets['text'], tokenizer, seq_length=140)
    # test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)
    # first_batch = next(iter(test_loader))
    # print(f'print first batch, batch_size = {first_batch["input_ids"].shape}')
    # print(first_batch)