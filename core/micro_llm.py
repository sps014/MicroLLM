import torch
import torch.nn as nn
import math
from torch.nn import functional as F
from dataclasses import dataclass
from typing import Optional, Tuple

# --- 1. CONFIGURATION ---
@dataclass(frozen=True)
class ModelConfig:
    """
    Defines the structural dimensions of the Transformer.
    Changing these values scales the model's capacity and memory footprint.
    """
    block_size: int = 512        # Maximum sequence length the model can process
    vocab_size: int = 50257      # Total number of unique tokens (GPT-2 standard)
    n_layer: int = 12            # Vertical depth: number of sequential Transformer blocks
    n_head: int = 12             # Horizontal width: number of parallel attention heads
    n_embd: int = 768            # Dimensionality of the latent space (C)
    dropout: float = 0.1         # Regularization to prevent overfitting by zeroing neurons
    bias: bool = False           # Modern LLMs omit bias in linear layers to improve scaling
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 2. ARCHITECTURE ---

class CausalSelfAttention(nn.Module):
    """
    The 'Communication' layer. Allows tokens to look back at previous tokens
    to understand context using the Query-Key-Value mechanism.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        # We fuse Q, K, and V into one large linear layer for computational efficiency.
        # This reduces the number of separate kernel calls on the GPU.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        
        # The projection layer merges the independent insights from multiple heads 
        # back into a single unified vector space.
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size() # Batch size, Sequence length, Embedding dim

        # Linear transformation to create Query, Key, and Value vectors.
        # split(dim=2) separates the fused output into three distinct tensors.
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        # We reshape tensors to separate the 'Heads'. 
        # This allows the model to attend to different types of information simultaneously.
        # Shape transition: (B, T, C) -> (B, nh, T, hs) where hs is head_size.
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Scaled Dot-Product Attention: The core math of the Transformer.
        # is_causal=True ensures tokens cannot see future tokens (masking).
        # We divide by sqrt(dk) internally to prevent gradient vanishing/explosion.
        y = F.scaled_dot_product_attention(
            q, k, v, 
            is_causal=True, 
            dropout_p=self.dropout if self.training else 0
        )
        
        # Re-assemble the parallel heads into a single sequence (Batch, Time, Channels).
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class Block(nn.Module):
    """
    A single Transformer layer. It consists of a communication phase (Attention)
    and a computation phase (MLP).
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        # LayerNorm is applied BEFORE the sub-layers (Pre-Norm) to keep 
        # signal variance stable across deep architectures.
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        
        # The MLP (Feed-Forward) expands the dimension to allow for complex 
        # feature extraction before projecting back to the original size.
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
            nn.GELU(), # Smoother non-linearity than ReLU
            nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual connections (x + ...) allow gradients to bypass layers during
        # backpropagation, effectively solving the vanishing gradient problem.
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class MicroLLM(nn.Module):
    """
    The top-level Transformer container.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            # Token Embedding converts integers into dense vectors.
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # Position Embedding provides the model with a sense of word order.
            wpe = nn.Embedding(config.block_size, config.n_embd),
            # The stack of Transformer blocks.
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        # The Language Model Head projects vectors back into Vocabulary space.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight Tying: Sharing weights between input and output layers 
        # regularizes the model and significantly reduces parameter count.
        self.transformer.wte.weight = self.lm_head.weight 

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        
        # Combine content information (wte) with spatial information (wpe).
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        
        for block in self.transformer.h:
            x = block(x)
            
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            # CrossEntropy expects (Batch*Time, Vocab) vs (Batch*Time).
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss
