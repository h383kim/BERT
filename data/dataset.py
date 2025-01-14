import torch
from torch import nn
from torch.utils.data import Dataset

import random

MAX_LEN = 100

def mask_tokens(input_seq, MARK_PROB=0.15):
    input_ids = tokenizer.encode(input_seq, truncation=True, max_length=MAX_LEN, add_special_tokens=False)

    segment_ids, labels = [], []
    is_second_sentence = False

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence1 = self.data.loc[idx, '원문']

        rand_idx = random.randint(0, len(self.data) - 1)
        sentence2 = self.data.loc[rand_idx, '원문']

        input_sequence = '[CLS]' + sentence1 + '[SEP]' + sentence2 + '[SEP]'

        nsp_label = torch.tensor(0)

        input_ids, mtp_label, nsp_label = mask_tokens(input_sequence)
        
        