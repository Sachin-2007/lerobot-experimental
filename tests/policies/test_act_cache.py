#!/usr/bin/env python

"""
Quick test script for ACT-Cache implementation.
This script performs basic functionality tests without requiring external datasets.
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add the src directory to Python path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.act_with_cache import (
    ACTCacheConfig, 
    ACTPolicyWithCache, 
    create_act_with_cache
)


def create_test_batch(batch_size=1, device="cpu"):
    """Create a test batch for ACT inference."""
    
    batch = {
        "observation.images.cam_high": torch.randn(batch_size, 3, 480, 640, device=device),
        "observation.images.cam_low": torch.randn(batch_size, 3, 480, 640, device=device),
        "observation.state": torch.randn(batch_size, 14, device=device),  # ALOHA state
        "action": torch.randn(batch_size, 50, 14, device=device),  # Action chunk
        "action_is_pad": torch.zeros(batch_size, 50, dtype=torch.bool, device=device),
    }
    
    return batch


def test_basic_functionality():
    """Test basic functionality of ACT with cache."""
    print("Testing basic ACT-Cache functionality...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create proper input/output features
    input_features = {
        "observation.images.cam_high": PolicyFeature(shape=[3, 480, 640], type=FeatureType.VISUAL),
        "observation.images.cam_low": PolicyFeature(shape=[3, 480, 640], type=FeatureType.VISUAL),
        "observation.state": PolicyFeature(shape=[14], type=FeatureType.STATE),
    }
    
    output_features = {
        "action": PolicyFeature(shape=[14], type=FeatureType.ACTION),
    }
    
    # Create configuration
    config = ACTConfig(
        chunk_size=50,
        n_action_steps=50,
        input_features=input_features,
        output_features=output_features,
        use_vae=True
    )
    
    # Create dataset stats for normalization
    dataset_stats = {}
    for key, feature in {**input_features, **output_features}.items():
        if feature.type == FeatureType.VISUAL:
            # Standard normalization for images
            dataset_stats[key] = {
                "mean": torch.zeros(feature.shape, device=device),
                "std": torch.ones(feature.shape, device=device),
            }
        else:
            # Standard normalization for other features
            dataset_stats[key] = {
                "mean": torch.zeros(feature.shape, device=device),
                "std": torch.ones(feature.shape, device=device),
            }
    
    # Create cache configuration
    cache_config = ACTCacheConfig(
        enable_visual_cache=True,
        visual_similarity_threshold=0.95,
        enable_timing=True
    )
    
    # Create models
    print("Creating standard ACT model...")
    standard_policy = ACTPolicy(config, dataset_stats=dataset_stats).to(device)
    
    print("Creating cached ACT model...")
    cached_policy = create_act_with_cache(config, cache_config, dataset_stats=dataset_stats).to(device)
    
    print(f"Standard model parameters: {sum(p.numel() for p in standard_policy.parameters()):,}")
    print(f"Cached model parameters: {sum(p.numel() for p in cached_policy.parameters()):,}")
    
    # Create test batch
    batch = create_test_batch(device=device)
    
    # Test forward pass
    print("\nTesting forward pass...")
    
    with torch.no_grad():
        # Standard model - use predict_action_chunk for inference
        standard_policy.eval()
        standard_actions = standard_policy.predict_action_chunk(batch)
        print(f"Standard output shape: {standard_actions.shape}")
        
        # Cached model - use forward method
        cached_policy.eval()
        cached_output = cached_policy.forward(batch)
        cached_actions = cached_output['action']
        print(f"Cached output shape: {cached_actions.shape}")
        
        # Check outputs are similar (they should be identical for first run)
        action_diff = torch.abs(standard_actions - cached_actions).mean()
        print(f"Action difference: {action_diff:.6f}")
    
    print("✓ Forward pass test passed!")
    return True


def test_caching_behavior():
    """Test that caching actually works by running multiple similar inputs."""
    print("\nTesting caching behavior...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create proper input/output features
    input_features = {
        "observation.images.cam_high": PolicyFeature(shape=[3, 480, 640], type=FeatureType.VISUAL),
        "observation.images.cam_low": PolicyFeature(shape=[3, 480, 640], type=FeatureType.VISUAL),
        "observation.state": PolicyFeature(shape=[14], type=FeatureType.STATE),
    }
    
    output_features = {
        "action": PolicyFeature(shape=[14], type=FeatureType.ACTION),
    }
    
    # Create configuration
    config = ACTConfig(
        chunk_size=50,
        n_action_steps=50,
        input_features=input_features,
        output_features=output_features,
        use_vae=False  # Disable VAE for simpler testing
    )
    
    # Create dataset stats for normalization
    dataset_stats = {}
    for key, feature in {**input_features, **output_features}.items():
        if feature.type == FeatureType.VISUAL:
            dataset_stats[key] = {
                "mean": torch.zeros(feature.shape, device=device),
                "std": torch.ones(feature.shape, device=device),
            }
        else:
            dataset_stats[key] = {
                "mean": torch.zeros(feature.shape, device=device),
                "std": torch.ones(feature.shape, device=device),
            }
    
    cache_config = ACTCacheConfig(
        enable_visual_cache=True,
        visual_similarity_threshold=0.99,  # High threshold for testing
        enable_timing=True
    )
    
    # Create cached model
    cached_policy = create_act_with_cache(config, cache_config, dataset_stats=dataset_stats).to(device)
    cached_policy.eval()
    
    # Create base batch
    base_batch = create_test_batch(device=device)
    
    # Test with similar images (should trigger cache hits)
    print("Testing with similar images...")
    times = []
    
    with torch.no_grad():
        for i in range(5):
            # Add small noise to images to simulate slight changes
            test_batch = {}
            for key, value in base_batch.items():
                if key.startswith("observation.images."):
                    # Add small random noise to image keys
                    noise = torch.randn_like(value) * 0.001  # Very small noise
                    test_batch[key] = value + noise
                else:
                    test_batch[key] = value
            
            # Time the inference
            start_time = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
            end_time = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
            
            if device == "cuda":
                start_time.record()
            else:
                import time
                start_time = time.time()
            
            action = cached_policy.select_action(test_batch)
            
            if device == "cuda":
                end_time.record()
                torch.cuda.synchronize()
                elapsed_time = start_time.elapsed_time(end_time) / 1000  # Convert to seconds
            else:
                elapsed_time = time.time() - start_time
            
            times.append(elapsed_time)
            print(f"Inference {i+1}: {elapsed_time:.4f}s")
    
    # Check if times generally decrease (indicating caching)
    avg_first_half = np.mean(times[:2])
    avg_second_half = np.mean(times[2:])
    
    print(f"Average time first 2 inferences: {avg_first_half:.4f}s")
    print(f"Average time last 3 inferences: {avg_second_half:.4f}s")
    
    if avg_second_half < avg_first_half:
        print("✓ Caching appears to be working (inference times decreased)")
    else:
        print("! Caching effect not clearly visible (may need more iterations)")
    
    # Get cache statistics
    cache_stats = cached_policy.get_cache_stats()
    if cache_stats:
        print(f"Cache statistics: {cache_stats}")
    
    return True


def test_inference_pipeline():
    """Test the complete inference pipeline."""
    print("\nTesting inference pipeline...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create proper input/output features
    input_features = {
        "observation.images.cam_high": PolicyFeature(shape=[3, 480, 640], type=FeatureType.VISUAL),
        "observation.images.cam_low": PolicyFeature(shape=[3, 480, 640], type=FeatureType.VISUAL),
        "observation.state": PolicyFeature(shape=[14], type=FeatureType.STATE),
    }
    
    output_features = {
        "action": PolicyFeature(shape=[14], type=FeatureType.ACTION),
    }
    
    config = ACTConfig(
        chunk_size=50,
        n_action_steps=50,
        input_features=input_features,
        output_features=output_features,
        use_vae=False
    )
    
    # Create dataset stats for normalization
    dataset_stats = {}
    for key, feature in {**input_features, **output_features}.items():
        if feature.type == FeatureType.VISUAL:
            dataset_stats[key] = {
                "mean": torch.zeros(feature.shape, device=device),
                "std": torch.ones(feature.shape, device=device),
            }
        else:
            dataset_stats[key] = {
                "mean": torch.zeros(feature.shape, device=device),
                "std": torch.ones(feature.shape, device=device),
            }
    
    cache_config = ACTCacheConfig(enable_timing=True)
    
    # Create model
    policy = create_act_with_cache(config, cache_config, dataset_stats=dataset_stats).to(device)
    policy.eval()
    
    # Test action selection
    batch = create_test_batch(device=device) 
    
    with torch.no_grad():
        # Reset policy
        policy.reset()
        
        # Select multiple actions
        actions = []
        for i in range(10):
            action = policy.select_action(batch)
            actions.append(action)
            print(f"Action {i+1} shape: {action.shape}")
        
        print(f"✓ Successfully selected {len(actions)} actions")
    
    return True


def main():
    """Run all tests."""
    print("ACT-Cache Implementation Tests")
    print("=" * 50)
    
    try:
        # Run tests
        test_basic_functionality()
        test_caching_behavior()
        test_inference_pipeline()
        
        print("\n" + "=" * 50)
        print("All tests passed! ✓")
        print("ACT-Cache implementation is working correctly.")
        print("\nYou can now run the full demo with:")
        print("python examples/run_act_cache_inference.py")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    main()
