#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
ACT-Cache Inference Demo

This script demonstrates how to use the ACT model with VLA-Cache inspired optimizations
for improved inference performance on ALOHA robot tasks.

Usage:
    python examples/run_act_cache_inference.py --policy_path <path_to_act_model> --dataset_path <path_to_test_data>
"""

import argparse
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.act_with_cache import (
    ACTCacheConfig,
    ACTPolicyWithCache,
    create_act_with_cache,
    visualize_cache_performance,
)


def load_dataset(dataset_path: str, batch_size: int = 1) -> DataLoader:
    """Load dataset for inference testing."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        dataset = LeRobotDataset(dataset_path)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        return dataloader
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Creating synthetic data for demonstration...")
        return create_synthetic_dataloader(batch_size)


def create_cube_transfer_synthetic_dataloader(batch_size: int = 1, num_samples: int = 50) -> DataLoader:
    """Create synthetic cube transfer data for demonstration when real dataset is not available."""
    
    class CubeTransferSyntheticDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples: int):
            self.num_samples = num_samples
            # Create some variation in the synthetic images to test caching with cube transfer patterns
            self.base_images = []
            for i in range(5):  # 5 different base images with cube transfer scenarios
                img = torch.randn(3, 480, 640) * 0.1 + (i * 0.2 - 0.4)
                # Add cube-like structure to simulate cube transfer task
                img[:, 200:280, 300:380] += 0.3  # Cube region
                img[:, 150:250, 250:350] += 0.2   # Robot arm region
                self.base_images.append(img)
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            # Use base images with small variations to simulate temporal consistency
            base_idx = idx % len(self.base_images)
            
            # Add small random variations - return tensors without extra batch dimension
            images = []
            for _ in range(2):  # Simulate 2 cameras
                img = self.base_images[base_idx] + torch.randn(3, 480, 640) * 0.01
                images.append(img)  # Remove the unsqueeze(0) - DataLoader adds batch dim
            
            # Generate cube transfer specific robot state and actions
            progress = (idx % 50) / 49.0  # Simulate progress through cube transfer
            
            # Robot state: positions and orientations for both arms (14 dims total)
            left_arm_state = torch.tensor([0.3 + progress * 0.2, 0.1, 0.05, 0.0, 0.0, 0.0, 1.0 - progress])  # 7 dims
            right_arm_state = torch.tensor([0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.5])  # 7 dims, stationary
            robot_state = torch.cat([left_arm_state, right_arm_state])
            
            # Generate cube transfer action sequence
            action_sequence = []
            for t in range(50):
                t_progress = t / 49.0
                # Left arm action progressing through cube transfer
                left_action = torch.tensor([
                    0.3 + t_progress * 0.2,  # x movement
                    0.1,                     # y position  
                    0.05 + t_progress * 0.02, # z lift
                    0.0, 0.0, 0.0,          # orientation
                    1.0 if t_progress < 0.2 or t_progress > 0.8 else 0.0  # gripper: open -> close -> open
                ])
                right_action = torch.tensor([0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.5])  # Right arm stationary
                action = torch.cat([left_action, right_action])
                action_sequence.append(action)
            
            actions = torch.stack(action_sequence)
            
            return {
                "observation.images.cam_high": images[0],
                "observation.images.cam_low": images[1], 
                "observation.state": robot_state,  # Remove batch dimension - DataLoader adds it
                "action": actions,  # Remove batch dimension - DataLoader adds it
                "action_is_pad": torch.zeros(50, dtype=torch.bool),  # Remove batch dimension
            }
    
    dataset = CubeTransferSyntheticDataset(num_samples)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def create_synthetic_dataloader(batch_size: int = 1, num_samples: int = 50) -> DataLoader:
    """Create synthetic data for demonstration when real dataset is not available."""
    # Use cube transfer data by default for more realistic manipulation testing
    return create_cube_transfer_synthetic_dataloader(batch_size, num_samples)


def benchmark_inference(
    policy: ACTPolicy,
    dataloader: DataLoader,
    num_steps: int = 100,
    description: str = "Standard ACT"
) -> Dict[str, float]:
    """Benchmark inference performance."""
    
    print(f"\n{'='*50}")
    print(f"Benchmarking {description}")
    print(f"{'='*50}")
    
    policy.eval()
    times = []
    actions_list = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_steps:
                break
                
            # Move batch to device
            device = next(policy.parameters()).device
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)
                elif isinstance(value, list):
                    batch[key] = [v.to(device) if isinstance(v, torch.Tensor) else v for v in value]
            
            # Reset policy state for each episode
            if i % 10 == 0:  # Reset every 10 steps to simulate episode boundaries
                policy.reset()
            
            # Time the inference
            start_time = time.time()
            action = policy.select_action(batch)
            end_time = time.time()
            
            inference_time = end_time - start_time
            times.append(inference_time)
            actions_list.append(action.cpu().numpy())
            
            if (i + 1) % 10 == 0:
                avg_time = np.mean(times[-10:])
                print(f"Step {i+1:3d}: {inference_time:.4f}s (avg: {avg_time:.4f}s)")
    
    # Calculate statistics
    times = np.array(times)
    actions_array = np.array(actions_list)
    
    stats = {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'fps': 1.0 / np.mean(times),
        'actions': actions_array
    }
    
    print(f"\n{description} Results:")
    print(f"  Mean inference time: {stats['mean_time']:.4f}s ± {stats['std_time']:.4f}s")
    print(f"  Min/Max time: {stats['min_time']:.4f}s / {stats['max_time']:.4f}s")
    print(f"  Inference FPS: {stats['fps']:.2f}")
    print(f"  Actions shape: {actions_array.shape}")
    
    return stats


def compare_accuracy(
    standard_actions: np.ndarray,
    cached_actions: np.ndarray,
    tolerance: float = 1e-6
) -> Dict[str, float]:
    """Compare accuracy between standard and cached model outputs."""
    
    print(f"\n{'='*50}")
    print("ACCURACY COMPARISON")
    print(f"{'='*50}")
    
    # Calculate various accuracy metrics
    absolute_diff = np.abs(standard_actions - cached_actions)
    relative_diff = np.abs(standard_actions - cached_actions) / (np.abs(standard_actions) + 1e-8)
    
    # Mean absolute error
    mae = np.mean(absolute_diff)
    
    # Root mean square error
    rmse = np.sqrt(np.mean(absolute_diff ** 2))
    
    # Maximum absolute error
    max_error = np.max(absolute_diff)
    
    # Percentage of actions within tolerance
    within_tolerance = np.mean(absolute_diff < tolerance) * 100
    
    # Cosine similarity for each action vector
    cosine_similarities = []
    for i in range(len(standard_actions)):
        std_action = standard_actions[i].flatten()
        cached_action = cached_actions[i].flatten()
        
        # Calculate cosine similarity
        dot_product = np.dot(std_action, cached_action)
        norm_std = np.linalg.norm(std_action)
        norm_cached = np.linalg.norm(cached_action)
        
        if norm_std > 0 and norm_cached > 0:
            cosine_sim = dot_product / (norm_std * norm_cached)
            cosine_similarities.append(cosine_sim)
    
    avg_cosine_similarity = np.mean(cosine_similarities) if cosine_similarities else 0.0
    
    # Correlation coefficient
    correlation = np.corrcoef(standard_actions.flatten(), cached_actions.flatten())[0, 1]
    
    accuracy_stats = {
        'mae': mae,
        'rmse': rmse,
        'max_error': max_error,
        'within_tolerance_percent': within_tolerance,
        'avg_cosine_similarity': avg_cosine_similarity,
        'correlation': correlation,
        'mean_absolute_diff': np.mean(absolute_diff),
        'std_absolute_diff': np.std(absolute_diff),
        'mean_relative_diff': np.mean(relative_diff),
        'std_relative_diff': np.std(relative_diff)
    }
    
    print(f"Action Shape: {standard_actions.shape}")
    print(f"Mean Absolute Error (MAE): {mae:.8f}")
    print(f"Root Mean Square Error (RMSE): {rmse:.8f}")
    print(f"Maximum Absolute Error: {max_error:.8f}")
    print(f"Actions within tolerance ({tolerance}): {within_tolerance:.2f}%")
    print(f"Average Cosine Similarity: {avg_cosine_similarity:.6f}")
    print(f"Correlation Coefficient: {correlation:.6f}")
    print(f"Mean Relative Difference: {np.mean(relative_diff):.6f} ± {np.std(relative_diff):.6f}")
    
    # Determine if models are equivalent
    is_equivalent = (mae < tolerance and avg_cosine_similarity > 0.999)
    print(f"\nModels are {'✓ EQUIVALENT' if is_equivalent else '✗ DIFFERENT'}")
    
    if not is_equivalent:
        print("⚠️  Warning: Cached model outputs differ significantly from standard model!")
        print("   This may indicate an issue with the caching implementation.")
    else:
        print("✅ Cached model maintains numerical accuracy!")
    
    return accuracy_stats


def compare_models(
    standard_policy: ACTPolicy,
    cached_policy: ACTPolicyWithCache,
    dataloader: DataLoader,
    num_steps: int = 100
):
    """Compare performance and accuracy between standard and cached ACT models."""
    
    print("\n" + "="*70)
    print("PERFORMANCE AND ACCURACY COMPARISON")
    print("="*70)
    
    # Benchmark standard model
    standard_stats = benchmark_inference(
        standard_policy, dataloader, num_steps, "Standard ACT"
    )
    
    # Benchmark cached model
    cached_stats = benchmark_inference(
        cached_policy, dataloader, num_steps, "ACT with Cache"
    )
    
    # Compare accuracy
    accuracy_stats = compare_accuracy(
        standard_stats['actions'], 
        cached_stats['actions'],
        tolerance=1e-6
    )
    
    # Calculate speedup
    speedup = standard_stats['mean_time'] / cached_stats['mean_time']
    fps_improvement = cached_stats['fps'] / standard_stats['fps']
    
    print(f"\n{'='*70}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*70}")
    print(f"Speedup: {speedup:.2f}x")
    print(f"FPS improvement: {fps_improvement:.2f}x")
    print(f"Time reduction: {(1 - 1/speedup)*100:.1f}%")
    
    # Get cache-specific statistics
    cache_stats = cached_policy.get_cache_stats()
    if cache_stats:
        print(f"\nCache Statistics:")
        print(f"  Average cache hit rate: {cache_stats.get('avg_cache_hit_rate', 0):.2%}")
        print(f"  Average backbone time: {cache_stats.get('avg_backbone_time', 0):.4f}s")
        print(f"  Total inferences: {cache_stats.get('total_inferences', 0)}")
    
    return {
        'standard': standard_stats,
        'cached': cached_stats,
        'accuracy': accuracy_stats,
        'speedup': speedup,
        'cache_stats': cache_stats
    }


def main():
    parser = argparse.ArgumentParser(description="ACT-Cache Inference Demo")
    parser.add_argument(
        "--policy_path", 
        type=str, 
        default=None,
        help="Path to pretrained ACT policy"
    )
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        default=None,
        help="Path to test dataset"
    )
    parser.add_argument(
        "--num_steps", 
        type=int, 
        default=100,
        help="Number of inference steps to benchmark"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=1,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on"
    )
    parser.add_argument(
        "--visualize", 
        action="store_true",
        help="Create performance visualization plots"
    )
    
    args = parser.parse_args()
    
    print("ACT-Cache Inference Demo")
    print("=" * 50)
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of steps: {args.num_steps}")
    
    # Set device
    device = torch.device(args.device)
    
    # Load or create dataset
    if args.dataset_path:
        print(f"Loading dataset from: {args.dataset_path}")
        dataloader = load_dataset(args.dataset_path, args.batch_size)
    else:
        print("Using synthetic dataset for demonstration")
        dataloader = create_synthetic_dataloader(args.batch_size)
    
    # Create ACT configuration with proper features
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
        use_vae=True
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
    
    # Create cache configuration with optimized settings
    cache_config = ACTCacheConfig(
        enable_visual_cache=True,
        visual_similarity_threshold=0.95,
        max_cached_features=10,
        enable_attention_cache=True,
        attention_reuse_threshold=0.8,
        enable_temporal_consistency=True,
        temporal_window_size=5,
        enable_timing=True,
        enable_visualization=args.visualize
    )
    
    # Create models
    print("\nCreating models...")
    
    if args.policy_path:
        # Load pretrained model
        print(f"Loading pretrained model from: {args.policy_path}")
        try:
            standard_policy = ACTPolicy.from_pretrained(args.policy_path)
            # Create cached version with same weights
            cached_policy = create_act_with_cache(standard_policy.config, cache_config)
            cached_policy.load_state_dict(standard_policy.state_dict(), strict=False)
        except Exception as e:
            print(f"Error loading pretrained model: {e}")
            print("Creating models from scratch...")
            standard_policy = ACTPolicy(config, dataset_stats=dataset_stats)
            cached_policy = create_act_with_cache(config, cache_config, dataset_stats=dataset_stats)
    else:
        # Create models from scratch
        print("Creating models from scratch...")
        standard_policy = ACTPolicy(config, dataset_stats=dataset_stats)
        cached_policy = create_act_with_cache(config, cache_config, dataset_stats=dataset_stats)
    
    # Move models to device
    standard_policy = standard_policy.to(device)
    cached_policy = cached_policy.to(device)
    
    print(f"Standard ACT parameters: {sum(p.numel() for p in standard_policy.parameters()):,}")
    print(f"Cached ACT parameters: {sum(p.numel() for p in cached_policy.parameters()):,}")
    
    # Run comparison
    results = compare_models(
        standard_policy, 
        cached_policy, 
        dataloader, 
        args.num_steps
    )
    
    # Create visualization if requested
    if args.visualize:
        print("\nCreating performance visualization...")
        viz_path = f"act_cache_performance_{int(time.time())}.png"
        try:
            visualize_cache_performance(cached_policy, viz_path)
        except Exception as e:
            print(f"Could not create visualization: {e}")
    
    print(f"\n{'='*70}")
    print("Demo completed successfully!")
    print(f"ACT-Cache achieved {results['speedup']:.2f}x speedup over standard ACT")
    
    if results['cache_stats']:
        hit_rate = results['cache_stats'].get('avg_cache_hit_rate', 0)
        print(f"Average cache hit rate: {hit_rate:.2%}")
    
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
