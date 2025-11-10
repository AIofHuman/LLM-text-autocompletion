import torch
import torch.nn as nn

class LSTMAutocomplete(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.3):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization with forget gate bias=1"""
        # Embedding
        nn.init.xavier_uniform_(self.embedding.weight)
        
        # LSTM
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                param.data.fill_(0)
                # Forget gate bias = 1
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)
        
        # Output layer
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            self.fc.bias.data.fill_(0)

    def forward(self, x, attention_mask=None, hidden=None):
        # x shape: (batch_size, seq_length)
        
        # Embedding
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        
        # LSTM
        lstm_out, hidden = self.lstm(embedded, hidden)  # lstm_out: (batch_size, seq_length, hidden_dim)
        
        # Use attention mask to find last non-padded token
        if attention_mask is not None:
            # Get the index of the last real token (not padding)
            lengths = attention_mask.sum(dim=1) - 1  # (batch_size,)
            # Use advanced indexing to get the last real output for each sequence
            last_output = lstm_out[torch.arange(lstm_out.size(0)), lengths]
        else:
            # Fallback to just taking the last position
            last_output = lstm_out[:, -1, :]
        
        # Apply dropout
        last_output = self.dropout_layer(last_output)
        
        # Final output
        output = self.fc(last_output)  # (batch_size, vocab_size)
        
        return output, hidden
    
    def generate_completion(self, tokenizer, input_sequence, max_target_length):
        """Generate list of tokens idx until <EOS> or max_target_length
        Args:
            tokinazer (Object): tokenizer which use for train modeld
            input_sequence (Tensor): tensor of tokens idx for predictions
            max_target_length (int): list of tokens idx - prediction for autocompletion
        Returns:
            Tesnsor(batch_size, max_target_length): text completion
        """
        result = torch.empty((input_sequence.size(0), max_target_length), dtype=torch.int32)
        if len(input_sequence) == 0:
            return result
        i = 0
        with torch.no_grad():
            while i < max_target_length:           
                output, _ = self.forward(input_sequence)
                pred = torch.argmax(output, dim=-1)
                # pred_token = tokenizer.decode(pred, skip_special_tokens=True) #tokenizer.convert_ids_to_tokens(pred)
                result[:, i] = pred
                input_sequence = self.replace_first_pad(input_sequence, pred)
                i+=1
        return result
    
    def replace_first_pad(self, input_sequence, new_tokens, pad_token_id=0):
        """
        Replaces the first pad token in each sequence with new tokens.

        Args:
            input_sequence: Tensor of shape (batch_size, seq_len)
            new_tokens: Tensor of shape (batch_size,) or (batch_size, k)
            pad_token_id: ID of the pad token

        Returns:
            Modified tensor
        """
        batch_size, seq_len = input_sequence.shape

        # Создаем маску pad-токенов
        pad_mask = (input_sequence == pad_token_id)

        # Находим позиции первого pad-токена в каждой последовательности
        first_pad_pos = pad_mask.int().argmax(dim=1)  # (batch_size,)

        # Создаем модифицированную последовательность
        modified_sequence = input_sequence.clone()

        for i in range(batch_size):
            pos = first_pad_pos[i]
            if pos < seq_len and pad_mask[i, pos]:
                if new_tokens.dim() == 1:
                    # Один токен для вставки
                    modified_sequence[i, pos] = new_tokens[i]
                else:
                    # Несколько токенов для вставки
                    k = new_tokens.shape[1]
                    if pos + k <= seq_len:
                        modified_sequence[i, pos:pos+k] = new_tokens[i]

        return modified_sequence

    
if __name__ == '__main__':
    import pandas as pd
    import os
    from tqdm import tqdm
    from pathlib import Path
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from transformers import BertTokenizerFast

    from next_token_dataset import NextTokenDataset, ValTokenDataset

    BASE_DIR = Path().resolve()
    MAX_LEN = 140
    device = 'cpu'

    
    # test run
    df_tweets = pd.read_csv(os.path.join(BASE_DIR, 'LLM-text-autocompletion','data', 'cleaned_tweets.csv'))
    # for test aim - only 100 samples
    df_tweets = df_tweets.sample(n=100).reset_index(drop=True)
    
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    train_dataset = NextTokenDataset(df_tweets['text'], tokenizer, seq_length=140)
    val_dataset = ValTokenDataset(df_tweets['text'], tokenizer, seq_length=140)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True)
    
    model = LSTMAutocomplete(tokenizer.vocab_size)
    criterion = nn.CrossEntropyLoss(ignore_index=0) 
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    epoch_train_loss = 0.0
    print('Run test epoch for train...')
    train_pbar = tqdm(train_loader, desc=f'Epoch 1 [Train]')
    for batch in train_pbar:
        input_ids = batch['input_ids'].to(device)
        targets = batch['target'].to(device)
        masks = batch['masks'].to(device)
        
        # Remove extra dimension from targets
        # targets = targets.squeeze()
        
        # Forward pass
        optimizer.zero_grad()
        outputs, hidden = model(input_ids, masks)
        
        # Calculate loss
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Statistics
        epoch_train_loss += loss.item()
    
    print(f'Loss: {epoch_train_loss/len(train_loader):.4f}')

    print('Test generate completion')
    model.eval()
    
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        targets = batch['target'].to(device)
        masks = batch['masks'].to(device)
        
        predicts, _ = model(input_ids, masks)

        # Remove extra dimension 
        # targets = targets.squeeze()
        # masks = masks.squeeze()
        # predict = predicts.squeeze()

        b_loss = criterion(predicts, targets[:,0])
        
        pred_texts = model.generate_completion(tokenizer, input_ids, max_target_length=MAX_LEN)

        target_text = [
            tokenizer.decode(targets[i], skip_special_tokens=True) 
            for i in range(targets.size(0))
        ]

        print('target texts')
        print(target_text)
        print('pred_texts')
        print(pred_texts)
        break