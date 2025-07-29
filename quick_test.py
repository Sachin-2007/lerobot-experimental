#!/usr/bin/env python

"""Quick test to verify ACT-Cache works."""

import sys
import torch
from pathlib import Path

# Add the src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.act_with_cache import ACTCacheConfig, create_act_with_cache

def main():
    print("Quick ACT-Cache test...")
    
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
        chunk_size=10,  # Small for testing
        n_action_steps=10,
        input_features=input_features,
        output_features=output_features,
        use_vae=False  # Disable VAE for simpler testing
    )
    
    # Create dataset stats
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
    
    # Create cached policy
    cache_config = ACTCacheConfig(enable_visual_cache=True, enable_timing=True)
    policy = create_act_with_cache(config, cache_config, dataset_stats=dataset_stats).to(device)
    policy.eval()
    
    # Create test batch
    batch = {
        "observation.images.cam_high": torch.randn(1, 3, 480, 640, device=device),
        "observation.images.cam_low": torch.randn(1, 3, 480, 640, device=device),
        "observation.state": torch.randn(1, 14, device=device),
        "action": torch.randn(1, 10, 14, device=device),
        "action_is_pad": torch.zeros(1, 10, dtype=torch.bool, device=device),
    }
    
    # Test forward pass
    print("Testing forward pass...")
    with torch.no_grad():
        output = policy.forward(batch)
        print(f"Output shape: {output['action'].shape}")
        
    # Test action selection
    print("Testing action selection...")
    with torch.no_grad():
        action = policy.select_action(batch)
        print(f"Action shape: {action.shape}")
    
    print("✓ Test passed!")

if __name__ == "__main__":
    main()
