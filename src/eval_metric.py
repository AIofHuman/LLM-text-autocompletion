from collections import Counter
import torch
import numpy as np
from tqdm import tqdm
import re

MAX_LEN = 140


def tokenize(text):
    """Токенизация текста"""
    return re.findall(r'\w+', text.lower())

def ngrams(tokens, n):
    """Создание n-грамм"""
    return zip(*[tokens[i:] for i in range(n)])

def rouge_n(references, candidates, n):
    """Расчет ROUGE-N"""
    precision_list = []
    recall_list = []
    f1_list = []
    
    for ref, cand in zip(references, candidates):
        ref_tokens = tokenize(ref)
        cand_tokens = tokenize(cand)
        
        ref_ngrams = list(ngrams(ref_tokens, n))
        cand_ngrams = list(ngrams(cand_tokens, n))
        
        ref_counter = Counter(ref_ngrams)
        cand_counter = Counter(cand_ngrams)
        
        # Пересечение n-грамм
        intersection = sum((ref_counter & cand_counter).values())
        
        # Расчет precision, recall, f1
        if len(cand_ngrams) == 0:
            precision = 0.0
        else:
            precision = intersection / len(cand_ngrams)
            
        if len(ref_ngrams) == 0:
            recall = 0.0
        else:
            recall = intersection / len(ref_ngrams)
            
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
            
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        
    return {
        'precision': sum(precision_list) / len(precision_list),
        'recall': sum(recall_list) / len(recall_list),
        'f1': sum(f1_list) / len(f1_list)
    }

def find_padding_start_np(sequence):
    arr = np.array(sequence.cpu().numpy())
    return np.argmax(arr == 0)

def id_tokens_to_text(tokenizer, tokens):        
    words = tokenizer.decode(tokens, skip_special_tokens=True)
    return ' '.join(words)

def rouge1_2(predictions, references):
    if len(predictions) == 0 or len(references) == 0:
        return 0, 0
    
    min_length = min(len(predictions), len(references))
    predictions = predictions[:min_length]
    references = references[:min_length]

    rouge1 = rouge_n(predictions, references, 1)['f1']
    rouge2 = rouge_n(predictions, references, 2)['f1']
    return rouge1, rouge2
          
def calc_metrics(model, loader, tokenizer,  device="cpu", criterion=None):
    model.eval()
    loss = 0.0
    rouge1 = 0.0
    rouge2 = 0.0

    val_pbar = tqdm(loader, desc=f'Calc metrics...')

    for i, batch in enumerate(val_pbar):
        input_ids = batch['input_ids'].to(device)
        targets = batch['target'].to(device)
        masks = batch['masks'].to(device)
        
        predicts, _ = model(input_ids, masks)

        # Remove extra dimension 
        # targets = targets.squeeze()
        # masks = masks.squeeze()
        # predict = predicts.squeeze()
        b_loss = 0.0
        if criterion != None:
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
            'rouge1': f'{rouge1/(i+1):.4f}  rouge2: {rouge2/(i+1):.4f}',
        })

    return {'loss': loss/len(loader), 'rouge1': rouge1/len(loader), 'rouge2': rouge2/len(loader)}