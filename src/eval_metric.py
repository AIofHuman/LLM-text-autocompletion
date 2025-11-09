import evaluate
import torch

# load metric
rouge = evaluate.load("rouge")


def id_tokens_to_text(tokenizer, tokens):        
        
        words = tokenizer.decode(tokens, skip_special_tokens=True)
        return ' '.join(words)

def generate_completion(model, vocab, input_sequence, input_mask, target_length, device):
        
        """Generate completion for the input sequence"""
        model.eval()
        generated_tokens = []
        
        with torch.no_grad():
            # Prepare initial input
            current_input = torch.tensor(input_sequence, device=device).unsqueeze(0)
            current_mask = torch.tensor(input_mask, device=device).unsqueeze(0) if input_mask is not None else None
            hidden = None
            
            for _ in range(target_length):
                # Forward pass
                output, hidden = model(current_input, current_mask, hidden)
                
                # Get next token (greedy decoding)
                next_token = torch.argmax(output, dim=-1)
                next_token_id = next_token.item()
                
                # Stop if EOS token
                if next_token_id == vocab.get('<EOS>', 3):
                    break
                
                # Add to generated tokens (skip special tokens)
                if next_token_id not in [vocab.get('<PAD>', 0), vocab.get('<SOS>', 2)]:
                    generated_tokens.append(next_token_id)
                
                # Update input for next step
                current_input = next_token.unsqueeze(0).unsqueeze(0)
                current_mask = torch.ones_like(current_input)
        
        return id_tokens_to_text(vocab, generated_tokens)

def rouge1_2(predictions, references):
    result = rouge.compute(predictions=predictions, references=references)
    return result['rouge1'], result['rouge2']
          
def calc_metrics(model, loader, criterion, tokenizer, device):
    model.eval()
    loss = 0
    rouge1 = 0
    rouge2 = 0
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        targets = batch['target'].to(device)
        masks = batch['masks'].to(device)
        
        predicts, _ = model(input_ids, masks)

        # Remove extra dimension 
        # targets = targets.squeeze()
        # masks = masks.squeeze()
        # predict = predicts.squeeze()

        b_loss = criterion(predicts, targets[:,0])
        
        # target_text = tokenizer.decode(targets, skip_special_tokens=True)
        # predict_text = tokenizer.decode(predict, skip_special_tokens=True)

        # b_rouge1, b_rouge2 = rouge1_2(predict_text, target_text)

        # rouge1 += b_rouge1
        # rouge2 += b_rouge2
        loss += b_loss

    return {'loss': loss/len(loader), 'rouge1': 0, 'rouge2': 0}#rouge1/len(loader), 'rouge2': rouge2/len(loader)}