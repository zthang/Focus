import argparse
from datasets import load_dataset
from collections import defaultdict
import pickle
import random
from transformers import AutoConfig, AutoTokenizer
import numpy as np
import math
from tqdm import tqdm


def calcu_idf(tokenizer_path):
    print(tokenizer_path)
    dataset = load_dataset("togethercomputer/RedPajama-Data-1T-Sample")
    data = [d for d in dataset["train"]]
    random.seed(42)
    random.shuffle(data)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    document_frequency = defaultdict(int)
    offset = 1 if 'facebook' in tokenizer_path else 0
    for doc in tqdm(data):
        tokenized_doc = tokenizer(doc["text"])["input_ids"][offset:]
        unique_tokens = set(tokenized_doc)
        for token in unique_tokens:
            document_frequency[token] += 1
    total_documents = len(data)
    pickle.dump(np.array([math.log(total_documents / (document_frequency[i] + 1)) for i in range(len(tokenizer.vocab))]), open(f"token_idf_{tokenizer_path.split('/')[-1]}.pkl", "wb"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, default="togethercomputer/RedPajama-INCITE-7B-Base")
    args = parser.parse_args()
    calcu_idf(args.tokenizer)

