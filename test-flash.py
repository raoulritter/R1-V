import torch
try:
    import flash_attn
    from flash_attn.flash_attn_interface import flash_attn_func
    
    # Check version
    print(f"Flash Attention version: {flash_attn.__version__}")
    
    # Create some test tensors
    batch_size, seq_len, n_head, d_head = 2, 1024, 8, 64
    q = torch.randn(batch_size, seq_len, n_head, d_head, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch_size, seq_len, n_head, d_head, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(batch_size, seq_len, n_head, d_head, device="cuda", dtype=torch.bfloat16)
    
    # Try to run flash attention
    out = flash_attn_func(q, k, v, causal=True)
    print("Flash Attention successfully executed!")
    
except ImportError as e:
    print(f"Import Error: {e}")
except Exception as e:
    print(f"Error during execution: {e}")