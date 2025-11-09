import re
import os 
import pandas as pd

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def clean_string(text: str) -> str:
    """ Convert input text to low register, delete special symbols, trim whitespaces

    Args:
        text (str): raw text

    Returns:
        str: text after processing
    """
    
    text = text.lower()
    # delete links
    text = re.sub(r'https?://\S+', '', text)
    # remind only alphafit and digits symbol
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # delete double whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # delete mentions
    text = re.sub(r'@\w+', '', text)
    
    return text
    

def dataset_processed(file_raw_dataset:str, file_processed_dataset: str):
    """ Read file with raw dataset, processed it and save it 
        to file_processed_datasets

    Args:
        file_raw_dataset (str): path to raw dataset file
        file_processed_dataset (str): path to processed dataset file
    """
    with open(file_raw_dataset, 'r', encoding='utf-8') as f:
        texts = f.readlines()

    cleaned_texts = list(map(clean_string, texts))

    df = pd.DataFrame({'text': cleaned_texts})
    df.to_csv(file_processed_dataset, index=False, encoding='utf-8')


if __name__ == '__main__':
    # test run
    dataset_processed(
        os.path.join(BASE_DIR, 'data', 'tweets.txt'),
        os.path.join(BASE_DIR, 'data', 'cleaned_tweets.csv')
    )
    print('clean_tweets datasets  were created!')