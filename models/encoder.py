import torch
from torch import nn

from models.embedding import BERTEmbedding
from models.multi_head_attention import MultiHeadAttention
from models.feed_forward_network import FFN


"""
Single block in the BERT encoder.

Each block consists of:
    - A multi-head self-attention mechanism with residual connections and LayerNorm.
    - A feed-forward network (FFN) with residual connections and LayerNorm.
    
Args:
    d_model (int): Dimensionality of the input embeddings.
    d_ffn (int): Dimensionality of the hidden layer in the feed-forward network.
    num_heads (int): Number of attention heads in the multi-head self-attention mechanism.
    p_dropout (float): Dropout probability for regularization.
"""
class EncoderBlock(nn.Module):
    def __init__(self, d_model, d_ffn, num_heads, p_dropout):
        super().__init__()
        # Self-attention
        self.norm1 = nn.LayerNorm()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.dropout1 = nn.Dropout(p=p_dropout)
        # Feed-Forward-Network
        self.norm2 = nn.LayerNorm()
        self.ffn = FFN()
        self.dropout2 = nn.Dropout(p=p_dropout)

    def forward(self, x, pad_mask):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            pad_mask (torch.Tensor): Mask tensor to prevent attention to specific positions.
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
            torch.Tensor: Attention scores from the self-attention mechanism.
        """
        # Self-attention and Skip_connect
        residual = self.norm1(x)
        residual, attn_score = self.self_attention(residual, residual, residual, mask=pad_mask)
        x = x + self.dropout1(residual) # norm -> attn -> dropout -> add
        # Feed_forward and Skip-connect
        residual = self.norm2(x)
        residual = self.ffn(residual)
        x = x + self.dropout2(residual)

        return x, attn_score



"""
Represents the full BERT encoder consisting of multiple encoder blocks.

The encoder applies:
    - Positional&segment encoding to the input embeddings.
    - Multiple encoder blocks, each with self-attention and feed-forward layers.

Args:
    vocab_size (int): Size of the vocabulary.
    max_len (int): Maximum sequence length.
    num_blocks (int): Number of encoder blocks in the encoder.
"""
class Encoder(nn.Module):
    def __init__(self, vocab_size, max_len, num_blocks, d_model, d_ffn, num_heads, p_dropout):
        super().__init__()
        # Create position&segment-encoded embedding
        self.input_emb = BERTEmbedding(vocab_size, max_len, d_model, p_dropout)
        self.dropout = nn.Dropout(p=p_dropout)
        # Create Encoder Blocks
        self.enc_blocks = nn.ModuleList([EncoderBlock(d_model, d_ffn, num_heads, p_dropout) 
                                         for _ in num_blocks])

        self.norm_out = nn.LayerNorm()

    def forward(self, x, seg, pad_mask, save_attn_pattern=False):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len) with token indices.
            pad_mask (torch.Tensor): Mask tensor to prevent attention to specific positions(e.g <PAD>).
            save_attn_pattern (bool): If True, saves attention patterns from each block for visualization.
        Returns:
            torch.Tensor: Final input tokens embedding tensor of shape (batch_size, seq_len, d_model).
            torch.Tensor: Attention patterns (if `save_attn_pattern` is True).
        """
        x = self.input_emb(x, seg)
        x = self.dropout(x)

        attn_patterns = torch.tensor([]).to(DEVICE)
        for block in self.enc_blocks:
            x = block(x, pad_mask)
            # (Optional) if save_attn_pattern is True, save these and return for visualization/investigation
            if save_attn_pattern:
                attn_patterns = torch.cat([attn_patterns, attn_pattern[0].unsqueeze(0)], dim=0)

        x = self.norm_out(x)
        
        return x, attn_patterns