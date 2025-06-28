import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from collections import Counter

# Vocabulary Class
class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold
    def __len__(self):  # ✅ Add this
        return len(self.stoi)

    # ✅ TEMPORARY TOKENIZER (No nltk, works on Python 3.13)
    def tokenizer(self, text):
        return text.lower().split()

    def build_vocab(self, sentence_list):
        frequencies = Counter()
        idx = 4

        for sentence in sentence_list:
            tokens = self.tokenizer(sentence)
            frequencies.update(tokens)

        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer(text)
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in tokenized_text]


# Dataset Class
class CaptionDataset(Dataset):
    def __init__(self, image_dir, captions_file, vocab, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.vocab = vocab
        self.imgs = []
        self.captions = []

        with open(captions_file, 'r', encoding='utf-8') as f:
            for line in f:
                if '\t' not in line:
                    print(f"Skipping invalid line (missing tab): {line.strip()}")
                    continue
                img, caption = line.strip().split('\t')
                self.imgs.append(img)
                self.captions.append(caption)

        self.vocab.build_vocab(self.captions)

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        caption = self.captions[index]
        img_id = self.imgs[index]
        img_path = os.path.join(self.image_dir, img_id)
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        return image, torch.tensor(numericalized_caption)
