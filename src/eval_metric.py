import evaluate
import torch
import numpy as np
from tqdm import tqdm

MAX_LEN = 140

# load metric
rouge = evaluate.load("rouge")

def find_padding_start_np(sequence):
    arr = np.array(sequence)
    return np.argmax(arr == 0)

def id_tokens_to_text(tokenizer, tokens):        
    words = tokenizer.decode(tokens, skip_special_tokens=True)
    return ' '.join(words)

# def generate_completion(model, vocab, input_sequence, input_mask, target_length, device):
        
#         """Generate completion for the input sequence"""
#         model.eval()
#         generated_tokens = []
        
#         with torch.no_grad():
#             # Prepare initial input
#             current_input = torch.tensor(input_sequence, device=device).unsqueeze(0)
#             current_mask = torch.tensor(input_mask, device=device).unsqueeze(0) if input_mask is not None else None
#             hidden = None
            
#             for _ in range(target_length):
#                 # Forward pass
#                 output, hidden = model(current_input, current_mask, hidden)
                
#                 # Get next token (greedy decoding)
#                 next_token = torch.argmax(output, dim=-1)
#                 next_token_id = next_token.item()
                
#                 # Stop if EOS token
#                 if next_token_id == vocab.get('<EOS>', 3):
#                     break
                
#                 # Add to generated tokens (skip special tokens)
#                 if next_token_id not in [vocab.get('<PAD>', 0), vocab.get('<SOS>', 2)]:
#                     generated_tokens.append(next_token_id)
                
#                 # Update input for next step
#                 current_input = next_token.unsqueeze(0).unsqueeze(0)
#                 current_mask = torch.ones_like(current_input)
        
#         return id_tokens_to_text(vocab, generated_tokens)

def rouge1_2(predictions, references):
    if len(predictions) == 0 or len(references) == 0:
        return 0, 0
    
    min_length = min(len(predictions), len(references))

    result = rouge.compute(
        predictions=predictions[:min_length],
        references=references[:min_length]
    )
    return result['rouge1'], result['rouge2']
          
def calc_metrics(model, loader, criterion, tokenizer, device):
    model.eval()
    loss = 0
    rouge1 = 0
    rouge2 = 0

    val_pbar = tqdm(loader, desc=f'Calc metrics...')

    for batch in val_pbar:
        input_ids = batch['input_ids'].to(device)
        targets = batch['target'].to(device)
        masks = batch['masks'].to(device)
        
        predicts, _ = model(input_ids, masks)

        # Remove extra dimension 
        # targets = targets.squeeze()
        # masks = masks.squeeze()
        # predict = predicts.squeeze()

        b_loss = criterion(predicts, targets[:,0])
        
        pred_idxs = model.generate_completion(tokenizer, input_ids, max_target_length=MAX_LEN)

        b_rouge1 = 0
        b_rouge2 = 0
        # loop into batch
        for pred, reference in zip(pred_idxs, targets):
            n_pad = find_padding_start_np(reference)
            reference = reference[:n_pad]
            pred = pred[:n_pad]
            
            text_reference = tokenizer.decode(reference, skip_special_tokens=True)
            text_pred = tokenizer.decode(pred, skip_special_tokens=True)
            
            # min_length = min(predictions.size(0), references.size(0))
            # predictions = predictions[:min_length]
            # references = references[:min_length]
            
            s_rouge1, s_rouge2 = rouge1_2(text_pred, text_reference)

            b_rouge1 += s_rouge1 
            b_rouge2 += s_rouge2
             

        # target_text = tokenizer.decode(targets, skip_special_tokens=True)
        # predict_text = tokenizer.decode(predict, skip_special_tokens=True)

        # b_rouge1, b_rouge2 = rouge1_2(predict_text, target_text)

        # rouge1 += b_rouge1
        # rouge2 += b_rouge2
        loss += b_loss
        rouge1 += b_rouge1/input_ids.shape[0]
        rouge2 += b_rouge2/input_ids.shape[0]
        # Update progress bar
        val_pbar.set_postfix({
            'rouge1': f'{rouge1:.4f}  rouge2: {rouge2:.4f}',
        })

    return {'loss': loss/len(loader), 'rouge1': rouge1/len(loader), 'rouge2': rouge2/len(loader)}