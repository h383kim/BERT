import torch
from torch import nn

from models.encoder import Encoder

"""
BERT model for pretraining(Masked_Token_Prediction + Next_Sentence_Prediction), 
consisting of encoder only from Transformer.

Args:
    pad_idx (int): Index of the <PAD> token in the vocabulary.
    vocab_size (int): Size of the vocabulary.
    max_len (int): Maximum length of sequences.
    num_blocks (int): Number of decoder blocks.
    d_model (int): Dimensionality of input embeddings and model representations.
    d_ffn (int): Dimensionality of the feed-forward network's hidden layer.
    num_heads (int): Number of attention heads in multi-head attention.
    p_dropout (float): Dropout probability for regularization.
       
Helper Methods:
    _make_pad_mask(src): Generates a pad token mask for the encoder's self-attention.
"""
class BERT(nn.Module):
    def __init__(self, pad_idx, vocab_size, max_len, num_blocks, d_model, d_ffn, num_heads, p_dropout):
        super().__init__()
        
        self.num_heads = num_heads
        self.pad_idx = pad_idx
        
        # Initialize encoder
        self.encoder = Encoder(vocab_size, max_len, num_blocks, d_model, d_ffn, num_heads, p_dropout)


    def forward(self, x, seg, save_attn_pattern=False):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len).
        Returns:
            tuple: A tuple containing:
                - encoder_out (torch.Tensor): Encoder output logits of shape (batch_size, seq_len, vocab_size).
                - attn_patterns (torch.Tensor): Self-attention patterns from the encoder.
        """
        # Create masking
        pad_mask = self._make_pad_mask(x)

        # Encoder pass
        encoder_out, attn_patterns = self.encoder(x, seg, pad_mask, save_attn_patterns)
        
        return encoder_out, attn_patterns

    def _make_pad_mask(self, x):
        """
        Creates a pad token mask for the encoder.
        Args:
            x (torch.Tensor): Target tensor of shape (batch_size, seq_len).
        Returns:
            pad_future_mask (torch.Tensor): Mask of shape (batch_size, num_heads, seq_len, seq_len), 
                                            where True indicates positions to mask (pad tokens).
        Example:
        (PAD MASK)
            Initial pad_mask for a single sequence (sentence): 
            [F F F T T] (F = False, T = True for <PAD>)

            Expanded across heads and queries(columns):
            [F F F T T]
            [F F F T T]
            [F F F T T]  x num_heads
            [F F F T T]
            [F F F T T]
        """
        pad_mask = (x == self.pad_idx).unsqueeze(1).unsqueeze(2)
        pad_mask = pad_mask.expand_as(x.shape[0], self.num_heads, x.shape[1], x.shape[1])
        return pad_mask



"""
BERT Language Model (BERTLM) for pretraining tasks.
This model implements two pretraining objectives:
1. Next Sentence Prediction (NSP): Predicts whether a segment follows the previous one.
2. Masked Token Prediction (MLM): Predicts the original tokens for masked positions.

Attributes:
    - bert (nn.Module): The core BERT model for encoding input sequences.
    - NextSentencePred (nn.Linear): A linear layer for next sentence prediction.
    - MaskedTokenPred (nn.Linear): A linear layer for masked token prediction.
Args:
    - bert (nn.Module): The core BERT model for encoding input sequences.
"""
class BERTLM(nn.Module):
    def __init__(self, bert, vocab_size, d_model):
        super().__init__()

        # BERT encoder model
        self.bert = bert

        # Linear layer for Next Sentence Prediction (NSP)
        # Input: [batch_size, d_model] -> Output: [batch_size, 2]
        self.NextSentencePred = nn.Linear(d_model, 2)

        # Linear layer for Masked Token Prediction (MLM)
        # Input: [batch_size, seq_len, d_model] -> Output: [batch_size, seq_len, vocab_size]
        self.MaskedTokenPred = nn.Linear(d_model, vocab_size)

    def forward(self, x, seg, save_attn_pattern=False):
        """
        Forward pass for the BERTLM model.
        Returns:
            - nsp (torch.Tensor): Next Sentence Prediction logits with shape [batch_size, 2].
            - mtp (torch.Tensor): Masked Token Prediction logits with shape [batch_size, seq_len, vocab_size].
            - attn_patterns (list, optional): Attention patterns from the BERT model, if requested.
        """
        # Pass inputs through the BERT model
        x, attn_patterns = self.bert(x, seg, save_attn_pattern)

        # NSP: Predict whether the second segment follows the first
        nsp = self.NextSentencePred(x[:, 0])  # Use the [CLS] token (x[:, 0]) for NSP

        # MLM: Predict the original tokens for masked positions
        mtp = self.MaskedTokenPred(x)  # Predict tokens for each position in the sequence

        return nsp, mtp, attn_patterns

        