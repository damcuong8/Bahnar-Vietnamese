"""
Script để debug lỗi lm_head với FSDP
"""
import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import os

# Mock config for testing
class MockConfig:
    def __init__(self):
        self.vocab_size = 256206
        self.hidden_size = 1024
        self.pad_token_id = 0
        self.tie_word_embeddings = True

def test_lm_head_basic():
    """Test basic lm_head functionality"""
    print("=" * 80)
    print("TEST 1: Basic lm_head test (no FSDP)")
    print("=" * 80)
    
    config = MockConfig()
    
    # Create components
    shared = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
    lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    
    print(f"Before tying:")
    print(f"  shared.weight shape: {shared.weight.shape}")
    print(f"  lm_head.weight shape: {lm_head.weight.shape}")
    print(f"  lm_head.bias: {lm_head.bias}")
    
    # Tie weights
    lm_head.weight = shared.weight
    
    print(f"\nAfter tying:")
    print(f"  lm_head.weight shape: {lm_head.weight.shape}")
    print(f"  lm_head.weight is shared.weight: {lm_head.weight is shared.weight}")
    print(f"  lm_head.bias: {lm_head.bias}")
    
    # Test forward pass
    batch_size = 4
    seq_len = 10
    test_input = torch.randn(batch_size, seq_len, config.hidden_size)
    
    print(f"\nTest input shape: {test_input.shape}")
    
    try:
        output = lm_head(test_input)
        print(f"✅ Output shape: {output.shape}")
        print(f"✅ Expected shape: ({batch_size}, {seq_len}, {config.vocab_size})")
    except Exception as e:
        print(f"❌ ERROR: {e}")
    
    print()


def test_lm_head_detach():
    """Test lm_head with detach and clone"""
    print("=" * 80)
    print("TEST 2: lm_head with detach/clone")
    print("=" * 80)
    
    config = MockConfig()
    
    shared = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
    lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    lm_head.weight = shared.weight
    
    batch_size = 4
    seq_len = 10
    test_input = torch.randn(batch_size, seq_len, config.hidden_size)
    
    print(f"Test input shape: {test_input.shape}")
    
    # Test 1: Just detach
    try:
        with torch.no_grad():
            output1 = lm_head(test_input).detach()
        print(f"✅ .detach() - Output shape: {output1.shape}")
    except Exception as e:
        print(f"❌ .detach() ERROR: {e}")
    
    # Test 2: Clone then detach
    try:
        with torch.no_grad():
            output2 = lm_head(test_input).clone().detach()
        print(f"✅ .clone().detach() - Output shape: {output2.shape}")
    except Exception as e:
        print(f"❌ .clone().detach() ERROR: {e}")
    
    # Test 3: Outside no_grad, then detach
    try:
        output3 = lm_head(test_input).detach()
        print(f"✅ Outside no_grad + .detach() - Output shape: {output3.shape}")
    except Exception as e:
        print(f"❌ Outside no_grad + .detach() ERROR: {e}")
    
    # Test 4: Outside no_grad, clone and detach
    try:
        output4 = lm_head(test_input).clone().detach()
        print(f"✅ Outside no_grad + .clone().detach() - Output shape: {output4.shape}")
    except Exception as e:
        print(f"❌ Outside no_grad + .clone().detach() ERROR: {e}")
    
    print()


def check_parameter_details():
    """Check detailed parameter information"""
    print("=" * 80)
    print("TEST 3: Parameter details inspection")
    print("=" * 80)
    
    config = MockConfig()
    
    shared = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
    lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    
    print(f"Before tying:")
    print(f"  shared.weight: shape={shared.weight.shape}, dtype={shared.weight.dtype}")
    print(f"  shared.weight: requires_grad={shared.weight.requires_grad}")
    print(f"  shared.weight: is_leaf={shared.weight.is_leaf}")
    print(f"  shared.weight: data_ptr={shared.weight.data_ptr()}")
    print()
    print(f"  lm_head.weight: shape={lm_head.weight.shape}, dtype={lm_head.weight.dtype}")
    print(f"  lm_head.weight: requires_grad={lm_head.weight.requires_grad}")
    print(f"  lm_head.weight: is_leaf={lm_head.weight.is_leaf}")
    print(f"  lm_head.weight: data_ptr={lm_head.weight.data_ptr()}")
    
    # Tie weights
    lm_head.weight = shared.weight
    
    print(f"\nAfter tying:")
    print(f"  lm_head.weight: shape={lm_head.weight.shape}, dtype={lm_head.weight.dtype}")
    print(f"  lm_head.weight: requires_grad={lm_head.weight.requires_grad}")
    print(f"  lm_head.weight: is_leaf={lm_head.weight.is_leaf}")
    print(f"  lm_head.weight: data_ptr={lm_head.weight.data_ptr()}")
    print(f"  Same data_ptr: {lm_head.weight.data_ptr() == shared.weight.data_ptr()}")
    print(f"  Same object: {lm_head.weight is shared.weight}")
    
    print()


if __name__ == "__main__":
    test_lm_head_basic()
    test_lm_head_detach()
    check_parameter_details()
    
    print("=" * 80)
    print("All tests completed!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Check the output above for any errors")
    print("2. Run your training script and compare the debug output")
    print("3. Look for differences in tensor shapes or states")
    print("4. Check if FSDP is changing the lm_head structure")
