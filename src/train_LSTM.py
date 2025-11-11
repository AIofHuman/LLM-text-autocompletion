import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
# import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path
from eval_metric import calc_metrics
import os 
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = 'models'


def train_model(model, train_loader, val_loader, tokenizer, device, num_epochs=10, learning_rate=0.001, save_dir='models'):
    """Train the LSTM autocomplete model"""
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding index
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # Training history
    train_losses = []
    val_losses = []
        
    print(f"Starting training on {device}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
                
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
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
            
            # Update progress bar
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
            })
        
        # Calculate training metrics
        avg_train_loss = epoch_train_loss / len(train_loader)
        metrics = calc_metrics(model, val_loader, tokenizer, device, criterion)
        
        # Update learning rate
        scheduler.step(avg_train_loss)
        
        # Store metrics
        train_losses.append(avg_train_loss)
        val_losses.append(metrics["loss"])
               
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Val loss: {metrics["loss"]:.4f}, rouge-1: {metrics["rouge1"]:.4f}, val rouge2: {metrics["rouge2"]:.4f}')

        save_final_model(model, optimizer, tokenizer, avg_train_loss, epoch)

def save_final_model(model, optimizer, tokenizer, final_train_loss, epoch):
    """Save the final model after training completes"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': final_train_loss,
    }
    
    # Save model
    torch.save(checkpoint, os.path.join(BASE_DIR, MODEL_DIR,'final_model.pth'))
    
    # Save tokenizer
    tokenizer.save_pretrained(os.path.join(BASE_DIR, MODEL_DIR,'tokenizer'))
    
    print("Model and tokenizer saved!")


        
if __name__ == '__main__':

    import pandas as pd
    import os
    from tqdm import tqdm
    from pathlib import Path
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from transformers import BertTokenizerFast

    from next_token_dataset import NextTokenDataset, ValTokenDataset
    from LSTM import LSTMAutocomplete

    BASE_DIR = Path().resolve()
    MAX_LEN = 140
    device = 'cpu'

    # test run
    df_tweets = pd.read_csv(os.path.join(BASE_DIR, 'LLM-text-autocompletion','data', 'cleaned_tweets.csv'))
    # for test aim - only 100 samples
    df_tweets = df_tweets.sample(n=100, random_state=42).reset_index(drop=True)
    # print(df_tweets.tail(10))
    
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    train_dataset = NextTokenDataset(df_tweets['text'], tokenizer, seq_length=140)
    val_dataset = ValTokenDataset(df_tweets['text'], tokenizer, seq_length=140)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True)
    
    model = LSTMAutocomplete(tokenizer.vocab_size)
    
    
    train_model(model, train_loader, val_loader, tokenizer, device, num_epochs=2, learning_rate=0.01, save_dir='models')
    