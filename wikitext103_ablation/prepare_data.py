import os
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

def prepare_wikitext103_data():
    print("Downloading WikiText-103 dataset (this may take a few minutes)...")
    # Using wikitext-103-raw-v1 for the larger raw text corpus
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    
    enc = tiktoken.get_encoding("gpt2")
    
    for split, filename in zip(['train', 'validation'], ['wiki_train.bin', 'wiki_val.bin']):
        print(f"Tokenizing {split} split...")
        
        # Extract text and drop empty lines
        texts = [t for t in dataset[split]['text'] if t.strip()]
        
        # Tokenize all text into a single flat list
        all_tokens = []
        for text in tqdm(texts, desc=f"Processing {split}"):
            all_tokens.extend(enc.encode_ordinary(text))
            
        print(f"Total tokens in {split}: {len(all_tokens):,}")
        
        # Save to binary format expected by np.memmap
        print(f"Saving to {filename}...")
        arr = np.array(all_tokens, dtype=np.uint16)
        arr.tofile(filename)
        
    print("Data preparation complete. wiki_train.bin and wiki_val.bin are ready.")

if __name__ == "__main__":
    prepare_wikitext103_data()