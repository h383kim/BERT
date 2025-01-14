import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import yaml
import pandas as pd
import random
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer

# Load configuration settings from a YAML file
with open("config/default.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)
    
MAX_LEN = 100


# Function to pad sequences to a specified maximum length
def pad_to_max_len(sequence, max_len, pad_value):
    return sequence + [pad_value] * (max_len - len(sequence))


# Function to mask tokens for pretraining
# This simulates the masking procedure used in BERT's Masked Language Model (MLM) objective
def mask_tokens(input_seq, tokenizer, MARK_PROB=0.15):
    CLS_ID = tokenizer.cls_token_id
    SEP_ID = tokenizer.sep_token_id
    PAD_ID = tokenizer.pad_token_id
    MASK_ID = tokenizer.mask_token_id

    # Tokenize the input sequence without special tokens
    input_ids = tokenizer.encode(
        input_seq, 
        truncation=True, 
        max_length=CONFIG["model"]["max_seq_len"], 
        add_special_tokens=False # Set as False as we will add [CLS] and [SEP] manually
    )

    ''' Initialize '''
    segment_ids, token_labels = [], []
    
    first_sentence = True
    for idx, token_id in enumerate(input_ids):
        ''' Segment Labeling (0 or 1) '''
        if token_id == SEP_ID: # If faced seperator of sentence1 and sentence2
            first_sentence = False
        segement_labels.append(0 if first sentence else 1)

        ''' Token Labeling and Masking '''
        random_mark = random.random()
        if (random_mark < MARK PROB) and (token not in (CLS_ID, SEP_ID, PAD_ID)):
            # Labels are the token id itself for those chosen to be marked by MARK_PROB%
            token_labels.append(token_id)

            # 80% are replaced with [MASK]
            # 10% with a random token, and 
            # 10% remain unchanged.
            random_mask = random.random()
            if random_mask < 0.8:
                input_ids[idx] = MASK_ID
            elif random_mask < 0.9:
                input_ids[idx] = random.randint(0, tokenizer.vocab_size - 1)
        
        # If not marked by 15% or is [CLS], [SEP], [PAD]
        # Assign ignore_idx label to opt out from loss calculation
        else:
            token_labels.append(CONFIG["model"]["ignore_idx"])

    # Pad sequences to the maximum length and Make them into Tensors
    input_ids = torch.tensor(pad_to_max_len(input_ids, CONFIG["model"]["max_seq_len"], PAD_ID))
    token_labels = torch.tensor(pad_to_max_len(token_labels, CONFIG["model"]["max_seq_len"], -100))
    segment_ids = torch.tensor(pad_to_max_len(segment_ids, CONFIG["model"]["max_seq_len"], 1))

    return input_ids, token_labels, segment_ids
            
      

class CustomDataset(Dataset):
    def __init__(self, data, tokenizer=None):
        self.data = data
        if tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")

    def __len__(self):
        # Retrieve the first sentence
        return len(self.data)

    def __getitem__(self, idx):
        sentence1 = self.data.loc[idx, '원문']

        # Randomly select a second sentence
        rand_idx = random.randint(0, len(self.data) - 1)
        sentence2 = self.data.loc[rand_idx, '원문']

        # Combine sentences with special tokens for next sentence prediction (NSP)
        input_sequence = '[CLS]' + sentence1 + '[SEP]' + sentence2 + '[SEP]'

        # NSP label is always 0 (not following sentence pairs for this dataset as we are randomly choosing second sentence)
        nsp_labels = torch.tensor(0)

        # Generate input IDs, token labels for MLM, and segment IDs
        input_ids, mtp_labels, segment_ids = mask_tokens(input_sequence, self.tokenizer)
        
        return input_ids, mtp_labels, nsp_labels, segment_idx


def get_dataloader(text_file_path, batch_size=CONFIG["train"]["batch_size"]):
    # Loading Data
    data = pd.read_excel(text_file_path, usecols=['번역문'])

    # Creating Dataset
    dataset = CustomDataset(data)

    # Spliiting into Train/Validation
    train_len = int(len(dataset) * 0.95)
    val_len = len(dataset) - train_len
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_len, val_len])

    # Creating DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader
