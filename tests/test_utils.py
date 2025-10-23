"""Test utilities for reproducibility and common operations."""
import random
import numpy as np
import torch
from grid_ai.utils import seed_everything, get_device


def test_seed_everything():
    """Test that seeding produces reproducible results across libraries."""
    seed = 42
    
    # First run
    seed_everything(seed)
    np_random1 = np.random.rand(5)
    torch_random1 = torch.rand(5)
    py_random1 = [random.random() for _ in range(5)]
    
    # Second run with same seed
    seed_everything(seed)
    np_random2 = np.random.rand(5)
    torch_random2 = torch.rand(5)
    py_random2 = [random.random() for _ in range(5)]
    
    # Check arrays are equal
    assert np.allclose(np_random1, np_random2)
    assert torch.allclose(torch_random1, torch_random2)
    assert py_random1 == py_random2


def test_get_device():
    """Test device selection logic."""
    # Test default behavior
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in ['cpu', 'cuda']
    
    # Test explicit CPU
    device = get_device('cpu')
    assert device.type == 'cpu'
    
    # Test CUDA (will fall back to CPU if not available)
    device = get_device('cuda')
    assert isinstance(device, torch.device)
    if not torch.cuda.is_available():
        assert device.type == 'cpu'
    else:
        assert device.type == 'cuda'