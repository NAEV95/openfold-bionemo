#!/usr/bin/env python3
"""
Debug the wrapper step by step
"""
import torch
import sys
import os
import importlib.util
import math

def debug_wrapper():
    """Debug wrapper step by step"""
    print("Debugging cuEquivariance wrapper...")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return False
    
    # Check if cuEquivariance is available
    cuequivariance_is_installed = importlib.util.find_spec("cuequivariance_torch") is not None
    if cuequivariance_is_installed:
        try:
            from cuequivariance_torch.primitives.triangle import triangle_attention
        except ImportError:
            cuequivariance_is_installed = False
    
    if not cuequivariance_is_installed:
        print("cuEquivariance not installed, skipping test")
        return False
    
    device = torch.device("cuda")
    
    # Test data
    num_heads = 2
    seq_len = 4
    hidden_dim = 8
    
    # Create CUDA tensors
    q = torch.randn(num_heads, seq_len, hidden_dim, device=device)
    k = torch.randn(num_heads, seq_len, hidden_dim, device=device)
    v = torch.randn(num_heads, seq_len, hidden_dim, device=device)
    bias = torch.randn(num_heads, seq_len, seq_len, device=device)
    mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
    
    print(f"Original shapes:")
    print(f"  q: {q.shape}")
    print(f"  k: {k.shape}")
    print(f"  v: {v.shape}")
    print(f"  bias: {bias.shape}")
    print(f"  mask: {mask.shape}")
    
    # Get original shape for reshaping
    orig_shape = q.shape
    batch_dims = orig_shape[:-3]
    
    print(f"batch_dims: {batch_dims}")
    
    # Reshape to [B, N, H, Q, D] format expected by cuEquivariance
    if len(batch_dims) == 0:
        # Add batch dimension if missing
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
        bias = bias.unsqueeze(0)
        if mask is not None:
            mask = mask.unsqueeze(0)
    
    print(f"After adding batch dimension:")
    print(f"  q: {q.shape}")
    print(f"  k: {k.shape}")
    print(f"  v: {v.shape}")
    print(f"  bias: {bias.shape}")
    print(f"  mask: {mask.shape}")
    
    # Transpose to match cuEquivariance format
    q = q.transpose(-2, -3)
    k = k.transpose(-2, -3)
    v = v.transpose(-2, -3)
    
    print(f"After transpose:")
    print(f"  q: {q.shape}")
    print(f"  k: {k.shape}")
    print(f"  v: {v.shape}")
    
    # For triangle attention, we need to expand to [B, N, H, Q, D]
    seq_len = q.shape[-3]
    q = q.unsqueeze(1).expand(-1, seq_len, -1, -1, -1).transpose(-2, -3)
    k = k.unsqueeze(1).expand(-1, seq_len, -1, -1, -1).transpose(-2, -3)
    v = v.unsqueeze(1).expand(-1, seq_len, -1, -1, -1).transpose(-2, -3)
    
    print(f"After expand and transpose:")
    print(f"  q: {q.shape}")
    print(f"  k: {k.shape}")
    print(f"  v: {v.shape}")
    
    # Adjust bias shape to [B, 1, H, Q, K]
    bias = bias.unsqueeze(1)
    
    print(f"After bias adjustment:")
    print(f"  bias: {bias.shape}")
    
    # Adjust mask shape to [B, N, 1, 1, K] if provided
    if mask is not None:
        mask = mask[:, :, -1].unsqueeze(1).expand(-1, seq_len, -1).unsqueeze(2).unsqueeze(3)
    
    print(f"After mask adjustment:")
    print(f"  mask: {mask.shape}")
    
    # Apply cuEquivariance triangle attention
    scale = 1.0 / math.sqrt(q.shape[-1])
    o = triangle_attention(
        q=q,
        k=k,
        v=v,
        bias=bias,
        mask=mask,
        scale=scale
    )
    
    print(f"After triangle_attention:")
    print(f"  o: {o.shape}")
    
    # Reshape back to original format
    o = o.squeeze(1)
    print(f"After squeeze(1):")
    print(f"  o: {o.shape}")
    
    o = o.transpose(-2, -3)
    print(f"After transpose(-2, -3):")
    print(f"  o: {o.shape}")
    
    # Remove batch dimension if it was added
    if len(batch_dims) == 0:
        o = o.squeeze(0)
        print(f"After squeeze(0):")
        print(f"  o: {o.shape}")
    
    # Expected output shape
    expected_shape = torch.Size([num_heads, seq_len, hidden_dim])
    print(f"Expected output shape: {expected_shape}")
    print(f"Actual output shape: {o.shape}")
    
    if o.shape == expected_shape:
        print("✓ Output shape is correct")
        return True
    else:
        print("✗ Output shape is incorrect")
        return False

if __name__ == "__main__":
    debug_wrapper()