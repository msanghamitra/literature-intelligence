# src/embeddings/pooling_strategies.py
# Pooling strategies for extracting sentence embeddings from transformer models.

from enum import Enum
from typing import Optional
import torch


class PoolingStrategy(Enum):
    """Available pooling strategies for extracting sentence embeddings."""
    MEAN = "mean"  # Mean pooling (default)
    CLS = "cls"    # CLS token pooling
    MAX = "max"    # Max pooling
    MEAN_MAX = "mean_max"  # Concatenated mean + max pooling
    WEIGHTED_MEAN = "weighted_mean"  # Attention-weighted mean pooling


def mean_pooling(
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Mean pooling strategy.
    
    Args:
        hidden_states: Last hidden states from transformer (batch_size, seq_len, hidden_dim)
        attention_mask: Attention mask (batch_size, seq_len)
        
    Returns:
        Pooled embeddings (batch_size, hidden_dim)
    """
    if attention_mask is None:
        # If no attention mask, use simple mean over sequence dimension
        return torch.mean(hidden_states, dim=1)
    
    # Expand attention mask for proper broadcasting
    attention_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    
    # Sum embeddings along sequence dimension
    sum_embeddings = torch.sum(hidden_states * attention_mask_expanded, dim=1)
    
    # Sum attention mask along sequence dimension
    sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
    
    # Return mean-pooled embeddings
    return sum_embeddings / sum_mask


def cls_pooling(hidden_states: torch.Tensor) -> torch.Tensor:
    """
    CLS token pooling strategy.
    
    Args:
        hidden_states: Last hidden states from transformer (batch_size, seq_len, hidden_dim)
        
    Returns:
        CLS token embeddings (batch_size, hidden_dim)
    """
    # Return the first token (CLS token) embeddings
    return hidden_states[:, 0]


def max_pooling(
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Max pooling strategy.
    
    Args:
        hidden_states: Last hidden states from transformer (batch_size, seq_len, hidden_dim)
        attention_mask: Attention mask (batch_size, seq_len)
        
    Returns:
        Max-pooled embeddings (batch_size, hidden_dim)
    """
    if attention_mask is None:
        # If no attention mask, use simple max over sequence dimension
        return torch.max(hidden_states, dim=1)[0]
    
    # Expand attention mask for proper broadcasting
    attention_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
    
    # Apply mask by setting masked positions to a very small value
    hidden_states_masked = hidden_states.clone()
    hidden_states_masked[~attention_mask_expanded.bool()] = -1e9
    
    # Return max over sequence dimension
    return torch.max(hidden_states_masked, dim=1)[0]


def mean_max_pooling(
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Concatenated mean + max pooling strategy.
    
    Args:
        hidden_states: Last hidden states from transformer (batch_size, seq_len, hidden_dim)
        attention_mask: Attention mask (batch_size, seq_len)
        
    Returns:
        Concatenated mean+max pooled embeddings (batch_size, hidden_dim * 2)
    """
    mean_pooled = mean_pooling(hidden_states, attention_mask)
    max_pooled = max_pooling(hidden_states, attention_mask)
    
    # Concatenate along the last dimension
    return torch.cat([mean_pooled, max_pooled], dim=-1)


def weighted_mean_pooling(
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Attention-weighted mean pooling strategy.
    
    Args:
        hidden_states: Last hidden states from transformer (batch_size, seq_len, hidden_dim)
        attention_mask: Attention mask (batch_size, seq_len)
        
    Returns:
        Weighted mean pooled embeddings (batch_size, hidden_dim)
    """
    if attention_mask is None:
        # If no attention mask, use simple mean
        return torch.mean(hidden_states, dim=1)
    
    # Use the last hidden state as attention weights
    # Simple approach: use mean of hidden states as attention
    weights = torch.mean(hidden_states, dim=-1, keepdim=True)  # (batch_size, seq_len, 1)
    
    # Apply softmax to get attention weights
    weights = torch.softmax(weights, dim=1)
    
    # Expand attention mask
    attention_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    
    # Apply mask to weights
    weights = weights * attention_mask_expanded
    
    # Normalize weights
    weights_sum = torch.clamp(torch.sum(weights, dim=1, keepdim=True), min=1e-9)
    weights = weights / weights_sum
    
    # Weighted sum
    weighted_sum = torch.sum(hidden_states * weights, dim=1)
    
    return weighted_sum


def apply_pooling_strategy(
    hidden_states: torch.Tensor,
    strategy: PoolingStrategy = PoolingStrategy.MEAN,
    attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Apply a pooling strategy to extract sentence embeddings.
    
    Args:
        hidden_states: Last hidden states from transformer (batch_size, seq_len, hidden_dim)
        strategy: Pooling strategy to use
        attention_mask: Attention mask (batch_size, seq_len)
        
    Returns:
        Pooled sentence embeddings
    """
    if strategy == PoolingStrategy.MEAN:
        return mean_pooling(hidden_states, attention_mask)
    elif strategy == PoolingStrategy.CLS:
        return cls_pooling(hidden_states)
    elif strategy == PoolingStrategy.MAX:
        return max_pooling(hidden_states, attention_mask)
    elif strategy == PoolingStrategy.MEAN_MAX:
        return mean_max_pooling(hidden_states, attention_mask)
    elif strategy == PoolingStrategy.WEIGHTED_MEAN:
        return weighted_mean_pooling(hidden_states, attention_mask)
    else:
        # Default to mean pooling
        return mean_pooling(hidden_states, attention_mask)


if __name__ == "__main__":
    # Test pooling strategies
    batch_size, seq_len, hidden_dim = 2, 10, 768
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
    attention_mask = torch.ones(batch_size, seq_len)
    attention_mask[0, 5:] = 0  # Mask second half of first sequence
    
    print("Testing pooling strategies:")
    print(f"Input shape: {hidden_states.shape}")
    
    for strategy in PoolingStrategy:
        pooled = apply_pooling_strategy(hidden_states, strategy, attention_mask)
        print(f"{strategy.value}: {pooled.shape}")