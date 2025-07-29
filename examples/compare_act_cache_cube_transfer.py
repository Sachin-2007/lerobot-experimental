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
ACT-Cache Cube Transfer Task Accuracy Comparison

This script compares the accuracy of ACT policy with and without VLA-Cache
on the ALOHA cube transfer task, providing detailed performance metrics.

Usage:
    python examples/compare_act_cache_cube_transfer.py [--pretrained_model <model_path>] [--num_episodes <num>]
"""

import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from huggingface_hub import hf_hub_download

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.act_with_cache import (
    ACTCacheConfig,
    ACTPolicyWithCache,
    create_act_with_cache,
    visualize_cache_performance,
)


def download_pretrained_cube_transfer_model() -> str:
    """Download the pretrained ACT model for cube transfer task."""
    print("Downloading pretrained ACT model for cube transfer task...")
    try:
        # Try to download the main model files
        model_path = hf_hub_download(
            repo_id="lerobot/act_aloha_sim_transfer_cube_human",
            filename="pytorch_model.bin",
            cache_dir="./models"
        )
        config_path = hf_hub_download(
            repo_id="lerobot/act_aloha_sim_transfer_cube_human", 
            filename="config.json",
            cache_dir="./models"
        )
        
        model_dir = str(Path(model_path).parent)
        print(f"Model downloaded to: {model_dir}")
        return model_dir
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("Will create model from scratch for demonstration...")
        return None


def load_cube_transfer_dataset(batch_size: int = 1) -> DataLoader:
    """Load the ALOHA cube transfer dataset."""
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        
        # Try different cube transfer datasets
        dataset_options = [
            "lerobot/aloha_sim_transfer_cube_human_image",
            "lerobot/aloha_sim_transfer_cube_human", 
            "lerobot/aloha_sim_transfer_cube_scripted_image",
            "lerobot/aloha_sim_transfer_cube_scripted"
        ]
        
        dataset = None
        dataset_name = None
        
        for repo_id in dataset_options:
            try:
                print(f"Trying to load dataset: {repo_id}")
                dataset = LeRobotDataset(repo_id)
                dataset_name = repo_id
                print(f"Successfully loaded dataset: {repo_id}")
                break
            except Exception as e:
                print(f"Failed to load {repo_id}: {e}")
                continue
        
        if dataset is None:
            raise ValueError("Could not load any cube transfer dataset")
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        print(f"Dataset loaded: {dataset_name}")
        print(f"Number of episodes: {len(dataset)}")
        
        return dataloader, dataset_name
        
    except Exception as e:
        print(f"Error loading real dataset: {e}")
        print("Creating cube transfer synthetic data for demonstration...")
        return create_cube_transfer_synthetic_data(batch_size), "synthetic"


def create_cube_transfer_synthetic_data(batch_size: int = 1, num_samples: int = 200) -> DataLoader:
    """Create synthetic data that mimics cube transfer task patterns."""
    
    class CubeTransferSyntheticDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples: int):
            self.num_samples = num_samples
            
            # Create cube transfer specific synthetic patterns
            # Simulate cube at different positions and orientations
            self.cube_positions = []
            self.target_positions = []
            
            for i in range(10):  # 10 different cube scenarios
                # Initial cube position (random but realistic) - 7 dims for ALOHA arm
                cube_pos = torch.tensor([
                    0.3 + (i % 3) * 0.1,  # x: 0.3 to 0.5
                    0.0 + (i % 2) * 0.1,  # y: 0.0 to 0.1
                    0.05,                 # z: on table
                    0.0, 0.0, i * 0.314,  # orientation (3 dims)
                    1.0,                  # gripper open initially
                ])
                
                # Target position (cube transfer destination) - 7 dims for ALOHA arm
                target_pos = torch.tensor([
                    0.0 + (i % 2) * 0.1,  # x: 0.0 to 0.1  
                    0.3 + (i % 3) * 0.1,  # y: 0.3 to 0.5
                    0.05,                 # z: on table
                    0.0, 0.0, (i+5) * 0.314,  # different orientation (3 dims)
                    1.0,                  # gripper open at end
                ])
                
                self.cube_positions.append(cube_pos)
                self.target_positions.append(target_pos)
            
            # Create base images that look like cube transfer scenarios
            self.scenario_images = []
            for i in range(5):
                # Simulate different cube transfer scenes
                img_high = torch.randn(3, 480, 640) * 0.1 + (i * 0.15 - 0.3)
                img_low = torch.randn(3, 480, 640) * 0.1 + (i * 0.12 - 0.25)
                
                # Add some structure to simulate cube/robot
                img_high[:, 200:280, 300:380] += 0.3  # Cube-like region
                img_low[:, 150:250, 250:350] += 0.2   # Robot arm region
                
                self.scenario_images.append((img_high, img_low))
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            # Select scenario based on index
            scenario_idx = idx % len(self.scenario_images)
            cube_idx = idx % len(self.cube_positions)
            
            # Get base images with small temporal variations
            img_high, img_low = self.scenario_images[scenario_idx]
            
            # Add small temporal noise (simulating camera jitter)
            img_high = img_high + torch.randn_like(img_high) * 0.01
            img_low = img_low + torch.randn_like(img_low) * 0.01
            
            # Get cube and robot state
            cube_state = self.cube_positions[cube_idx]
            target_state = self.target_positions[cube_idx]
            
            # Combined robot state (14-dim for ALOHA: 7 per arm)
            robot_state = torch.cat([
                cube_state,     # Current left arm pose/gripper state (7 dims)
                target_state,   # Right arm or target info (7 dims)  
            ])
            
            # Generate realistic cube transfer actions (7 dims per arm = 14 total)
            # Action sequence for cube transfer: approach, grasp, lift, move, place
            action_sequence = []
            for t in range(50):  # 50 timesteps
                progress = t / 49.0  # 0 to 1
                
                # Left arm: Interpolate between current and target positions
                action_left = cube_state + progress * (target_state - cube_state)
                
                # Right arm: Keep stationary
                action_right = torch.tensor([0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.5])  # 7 dims with gripper half-closed
                
                # Modify left gripper based on task phase
                if progress < 0.2:  # Approach
                    action_left[6] = 1.0  # Open gripper
                elif progress < 0.4:  # Grasp
                    action_left[6] = 0.0  # Close gripper  
                elif progress < 0.8:  # Move
                    action_left[6] = 0.0  # Keep gripper closed
                else:  # Place
                    action_left[6] = 1.0  # Open gripper
                
                action = torch.cat([action_left, action_right])  # Combine to 14 dims
                action_sequence.append(action)
            
            actions = torch.stack(action_sequence)
            
            return {
                "observation.images.cam_high": img_high,
                "observation.images.cam_low": img_low,
                "observation.state": robot_state,
                "action": actions,
                "action_is_pad": torch.zeros(50, dtype=torch.bool),
            }
    
    dataset = CubeTransferSyntheticDataset(num_samples)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def create_cube_transfer_config() -> ACTConfig:
    """Create ACT configuration for cube transfer task."""
    
    # Define input and output features for ALOHA cube transfer
    input_features = {
        "observation.images.cam_high": PolicyFeature(shape=[3, 480, 640], type=FeatureType.VISUAL),
        "observation.images.cam_low": PolicyFeature(shape=[3, 480, 640], type=FeatureType.VISUAL),
        "observation.state": PolicyFeature(shape=[14], type=FeatureType.STATE),  # 14-dim robot state
    }
    
    output_features = {
        "action": PolicyFeature(shape=[14], type=FeatureType.ACTION),  # 14-dim action (7 per arm)
    }
    
    # Configuration optimized for cube transfer task
    config = ACTConfig(
        chunk_size=100,          # Longer sequences for manipulation
        n_action_steps=100,      # Match chunk size
        input_features=input_features,
        output_features=output_features,
        use_vae=True,
        latent_dim=32,           # Reasonable latent space
        n_heads=8,               # Attention heads
        dim_feedforward=3200,    # Feedforward dimension
        n_decoder_layers=7,      # Decoder layers
        dropout=0.1,
        kl_weight=10.0,         # KL divergence weight
    )
    
    return config


def benchmark_cube_transfer_accuracy(
    policy: ACTPolicy,
    dataloader: DataLoader,
    num_episodes: int = 50,
    description: str = "ACT Policy"
) -> Dict[str, any]:
    """Benchmark policy performance on cube transfer task."""
    
    print(f"\n{'='*60}")
    print(f"Benchmarking {description} on Cube Transfer Task")
    print(f"{'='*60}")
    
    policy.eval()
    episode_times = []
    all_actions = []
    success_metrics = []
    
    with torch.no_grad():
        for episode in range(min(num_episodes, len(dataloader))):
            episode_start = time.time()
            
            episode_actions = []
            episode_states = []
            
            # Get episode data
            batch = next(iter(dataloader))
            
            # Move to device
            device = next(policy.parameters()).device
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)
            
            # Reset policy for new episode
            policy.reset()
            
            # Run episode
            step_times = []
            for step in range(batch["action"].shape[1]):  # Action sequence length
                step_start = time.time()
                
                # Create observation for this step
                obs = {
                    "observation.images.cam_high": batch["observation.images.cam_high"],
                    "observation.images.cam_low": batch["observation.images.cam_low"],
                    "observation.state": batch["observation.state"],
                }
                
                # Get action
                action = policy.select_action(obs)
                episode_actions.append(action.cpu().numpy())
                
                step_end = time.time()
                step_times.append(step_end - step_start)
            
            episode_end = time.time()
            episode_time = episode_end - episode_start
            episode_times.append(episode_time)
            
            # Store actions for this episode
            episode_actions_array = np.array(episode_actions)
            all_actions.append(episode_actions_array)
            
            # Calculate success metrics (simplified for synthetic data)
            final_action = episode_actions_array[-1]
            success_score = calculate_cube_transfer_success(final_action)
            success_metrics.append(success_score)
            
            if (episode + 1) % 10 == 0:
                avg_time = np.mean(episode_times[-10:])
                avg_success = np.mean(success_metrics[-10:])
                print(f"Episode {episode+1:3d}: {episode_time:.3f}s (avg: {avg_time:.3f}s) | Success: {success_score:.3f} (avg: {avg_success:.3f})")
    
    # Calculate statistics
    episode_times = np.array(episode_times)
    all_actions = np.array(all_actions)
    success_metrics = np.array(success_metrics)
    
    stats = {
        'mean_episode_time': np.mean(episode_times),
        'std_episode_time': np.std(episode_times),
        'mean_step_time': np.mean(episode_times) / batch["action"].shape[1],
        'success_rate': np.mean(success_metrics),
        'success_std': np.std(success_metrics),
        'actions': all_actions,
        'episodes_fps': 1.0 / np.mean(episode_times),
        'total_episodes': len(episode_times)
    }
    
    print(f"\n{description} Results:")
    print(f"  Episodes completed: {stats['total_episodes']}")
    print(f"  Mean episode time: {stats['mean_episode_time']:.3f}s ± {stats['std_episode_time']:.3f}s")
    print(f"  Mean step time: {stats['mean_step_time']:.4f}s")
    print(f"  Episode FPS: {stats['episodes_fps']:.2f}")
    print(f"  Success rate: {stats['success_rate']:.3f} ± {stats['success_std']:.3f}")
    print(f"  Actions shape: {all_actions.shape}")
    
    return stats


def calculate_cube_transfer_success(final_action: np.ndarray) -> float:
    """Calculate success metric for cube transfer task."""
    # Simplified success metric based on final action
    # In real evaluation, this would check if cube reached target position
    
    # Check if gripper is in reasonable position and open (indicating successful place)
    left_arm = final_action[:7]
    gripper_pos = left_arm[:3]  # x, y, z position
    gripper_state = left_arm[6]  # gripper open/close
    
    # Success criteria (simplified):
    # 1. Gripper in reasonable target area
    # 2. Gripper is open (released cube)
    pos_score = np.exp(-np.linalg.norm(gripper_pos - np.array([0.1, 0.4, 0.05])))
    grip_score = gripper_state  # Higher values mean more open
    
    success_score = 0.7 * pos_score + 0.3 * grip_score
    return min(success_score, 1.0)


def compare_cube_transfer_accuracy(
    standard_actions: np.ndarray,
    cached_actions: np.ndarray,
    tolerance: float = 1e-5
) -> Dict[str, float]:
    """Compare accuracy between standard and cached models on cube transfer task."""
    
    print(f"\n{'='*60}")
    print("CUBE TRANSFER ACCURACY COMPARISON")
    print(f"{'='*60}")
    
    # Flatten for easier analysis
    std_flat = standard_actions.reshape(-1, standard_actions.shape[-1])
    cached_flat = cached_actions.reshape(-1, cached_actions.shape[-1])
    
    # Calculate detailed accuracy metrics
    absolute_diff = np.abs(std_flat - cached_flat)
    relative_diff = np.abs(std_flat - cached_flat) / (np.abs(std_flat) + 1e-8)
    
    # Overall metrics
    mae = np.mean(absolute_diff)
    rmse = np.sqrt(np.mean(absolute_diff ** 2))
    max_error = np.max(absolute_diff)
    within_tolerance = np.mean(absolute_diff < tolerance) * 100
    
    # Per-action-dimension analysis (important for manipulation)
    per_dim_mae = np.mean(absolute_diff, axis=0)
    per_dim_max = np.max(absolute_diff, axis=0)
    
    # Manipulation-specific metrics
    left_arm_mae = np.mean(per_dim_mae[:7])   # Left arm (primary manipulator)
    right_arm_mae = np.mean(per_dim_mae[7:])  # Right arm 
    gripper_mae = np.mean([per_dim_mae[6], per_dim_mae[13]])  # Both grippers
    
    # Temporal consistency (important for smooth manipulation)
    temporal_consistency = calculate_temporal_consistency(std_flat, cached_flat)
    
    # Position vs orientation accuracy
    pos_mae = np.mean([per_dim_mae[i] for i in [0, 1, 2, 7, 8, 9]])      # Positions
    orient_mae = np.mean([per_dim_mae[i] for i in [3, 4, 5, 10, 11, 12]]) # Orientations
    
    # Correlation analysis
    correlation = np.corrcoef(std_flat.flatten(), cached_flat.flatten())[0, 1]
    
    accuracy_stats = {
        'mae': mae,
        'rmse': rmse,
        'max_error': max_error,
        'within_tolerance_percent': within_tolerance,
        'correlation': correlation,
        'left_arm_mae': left_arm_mae,
        'right_arm_mae': right_arm_mae,
        'gripper_mae': gripper_mae,
        'position_mae': pos_mae,
        'orientation_mae': orient_mae,
        'temporal_consistency': temporal_consistency,
        'per_dim_mae': per_dim_mae,
        'per_dim_max': per_dim_max
    }
    
    print(f"Overall Accuracy Metrics:")
    print(f"  Mean Absolute Error: {mae:.6f}")
    print(f"  Root Mean Square Error: {rmse:.6f}")
    print(f"  Maximum Error: {max_error:.6f}")
    print(f"  Within tolerance ({tolerance}): {within_tolerance:.2f}%")
    print(f"  Correlation: {correlation:.6f}")
    
    print(f"\nManipulation-Specific Metrics:")
    print(f"  Left Arm MAE: {left_arm_mae:.6f}")
    print(f"  Right Arm MAE: {right_arm_mae:.6f}")
    print(f"  Gripper MAE: {gripper_mae:.6f}")
    print(f"  Position MAE: {pos_mae:.6f}")
    print(f"  Orientation MAE: {orient_mae:.6f}")
    print(f"  Temporal Consistency: {temporal_consistency:.6f}")
    
    print(f"\nPer-Dimension Analysis:")
    dim_names = ['LX', 'LY', 'LZ', 'LRx', 'LRy', 'LRz', 'LGrip', 'RX', 'RY', 'RZ', 'RRx', 'RRy', 'RRz', 'RGrip']
    for i, (name, mae_val, max_val) in enumerate(zip(dim_names, per_dim_mae, per_dim_max)):
        print(f"  {name:>5}: MAE={mae_val:.6f}, Max={max_val:.6f}")
    
    # Determine if models are equivalent for manipulation tasks
    is_equivalent = (mae < tolerance and 
                    temporal_consistency > 0.95 and
                    gripper_mae < tolerance * 2)  # Grippers are critical
    
    print(f"\nCube Transfer Equivalence Assessment:")
    print(f"Models are {'✓ EQUIVALENT' if is_equivalent else '✗ DIFFERENT'} for cube transfer task")
    
    if not is_equivalent:
        print("⚠️  Warning: Cached model may affect cube transfer performance!")
        if gripper_mae > tolerance * 2:
            print("   → Gripper control accuracy is concerning")
        if temporal_consistency < 0.95:
            print("   → Temporal consistency may cause jerky movements")
    else:
        print("✅ Cached model maintains manipulation accuracy!")
    
    return accuracy_stats


def calculate_temporal_consistency(std_actions: np.ndarray, cached_actions: np.ndarray) -> float:
    """Calculate temporal consistency between action sequences."""
    # Calculate smoothness of action differences
    std_diff = np.diff(std_actions, axis=0)
    cached_diff = np.diff(cached_actions, axis=0)
    
    # Compare the derivatives (velocity consistency)
    velocity_similarity = 1.0 - np.mean(np.abs(std_diff - cached_diff))
    return max(0.0, min(1.0, velocity_similarity))


def main():
    parser = argparse.ArgumentParser(description="ACT-Cache Cube Transfer Accuracy Comparison")
    parser.add_argument(
        "--pretrained_model", 
        type=str, 
        default=None,
        help="Path to pretrained ACT model (will download if not provided)"
    )
    parser.add_argument(
        "--num_episodes", 
        type=int, 
        default=50,
        help="Number of episodes to test"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on"
    )
    parser.add_argument(
        "--visualize", 
        action="store_true",
        help="Create performance visualization"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-5,
        help="Tolerance for accuracy comparison"
    )
    
    args = parser.parse_args()
    
    print("ACT-Cache Cube Transfer Accuracy Comparison")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Tolerance: {args.tolerance}")
    
    device = torch.device(args.device)
    
    # Load cube transfer dataset
    print("\nLoading cube transfer dataset...")
    try:
        dataloader, dataset_name = load_cube_transfer_dataset(batch_size=1)
        print(f"Dataset: {dataset_name}")
    except:
        dataloader = create_cube_transfer_synthetic_data(batch_size=1)
        dataset_name = "synthetic_cube_transfer"
        print(f"Using synthetic cube transfer data")
    
    # Create or load model
    config = create_cube_transfer_config()
    
    if args.pretrained_model:
        model_path = args.pretrained_model
    else:
        model_path = download_pretrained_cube_transfer_model()
    
    print("\nCreating models...")
    if model_path:
        try:
            print(f"Loading pretrained model from: {model_path}")
            standard_policy = ACTPolicy.from_pretrained(model_path)
        except Exception as e:
            print(f"Error loading pretrained model: {e}")
            print("Creating model from scratch...")
            standard_policy = ACTPolicy(config)
    else:
        print("Creating model from scratch...")
        standard_policy = ACTPolicy(config)
    
    # Create cached version
    cache_config = ACTCacheConfig(
        enable_visual_cache=True,
        visual_similarity_threshold=0.97,  # Slightly higher for manipulation precision
        max_cached_features=15,
        enable_attention_cache=True,
        attention_reuse_threshold=0.85,
        enable_temporal_consistency=True,
        temporal_window_size=8,           # Longer window for manipulation
        enable_timing=True,
    )
    
    cached_policy = create_act_with_cache(config, cache_config)
    
    # Copy weights if we have a pretrained model
    if model_path:
        try:
            cached_policy.load_state_dict(standard_policy.state_dict(), strict=False)
        except Exception as e:
            print(f"Warning: Could not copy weights: {e}")
    
    # Move to device
    standard_policy = standard_policy.to(device)
    cached_policy = cached_policy.to(device)
    
    print(f"Standard ACT parameters: {sum(p.numel() for p in standard_policy.parameters()):,}")
    print(f"Cached ACT parameters: {sum(p.numel() for p in cached_policy.parameters()):,}")
    
    # Run benchmarks
    print("\n" + "="*80)
    print("CUBE TRANSFER TASK EVALUATION")
    print("="*80)
    
    # Benchmark standard model
    standard_stats = benchmark_cube_transfer_accuracy(
        standard_policy, dataloader, args.num_episodes, "Standard ACT"
    )
    
    # Benchmark cached model  
    cached_stats = benchmark_cube_transfer_accuracy(
        cached_policy, dataloader, args.num_episodes, "ACT with VLA-Cache"
    )
    
    # Compare accuracy
    accuracy_stats = compare_cube_transfer_accuracy(
        standard_stats['actions'], 
        cached_stats['actions'],
        tolerance=args.tolerance
    )
    
    # Performance comparison
    speedup = standard_stats['mean_episode_time'] / cached_stats['mean_episode_time']
    success_diff = cached_stats['success_rate'] - standard_stats['success_rate']
    
    print(f"\n{'='*80}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"Performance:")
    print(f"  Episode Speedup: {speedup:.2f}x")
    print(f"  Standard FPS: {standard_stats['episodes_fps']:.2f}")
    print(f"  Cached FPS: {cached_stats['episodes_fps']:.2f}")
    
    print(f"\nTask Performance:")
    print(f"  Standard Success Rate: {standard_stats['success_rate']:.3f}")
    print(f"  Cached Success Rate: {cached_stats['success_rate']:.3f}")
    print(f"  Success Rate Difference: {success_diff:+.3f}")
    
    print(f"\nAccuracy:")
    print(f"  Overall MAE: {accuracy_stats['mae']:.6f}")
    print(f"  Manipulation Critical MAE: {accuracy_stats['gripper_mae']:.6f}")
    print(f"  Temporal Consistency: {accuracy_stats['temporal_consistency']:.6f}")
    
    # Cache statistics
    cache_stats = cached_policy.get_cache_stats()
    if cache_stats:
        print(f"\nCache Performance:")
        print(f"  Hit Rate: {cache_stats.get('avg_cache_hit_rate', 0):.2%}")
        print(f"  Backbone Time Saved: {cache_stats.get('avg_backbone_time', 0):.4f}s per step")
    
    # Final assessment
    manipulation_safe = (accuracy_stats['gripper_mae'] < args.tolerance * 2 and 
                        accuracy_stats['temporal_consistency'] > 0.95)
    
    print(f"\n{'='*80}")
    print("CUBE TRANSFER TASK ASSESSMENT")
    print(f"{'='*80}")
    if manipulation_safe:
        print("✅ VLA-Cache is SAFE for cube transfer task")
        print(f"   → {speedup:.1f}x speedup with maintained accuracy")
        print(f"   → Success rate change: {success_diff:+.1%}")
    else:
        print("⚠️  VLA-Cache may IMPACT cube transfer performance")
        print("   → Consider tighter similarity thresholds")
        print("   → Monitor gripper control carefully")
    
    # Visualization
    if args.visualize:
        print("\nCreating visualization...")
        try:
            viz_path = f"cube_transfer_comparison_{int(time.time())}.png"
            visualize_cache_performance(cached_policy, viz_path)
            print(f"Visualization saved to: {viz_path}")
        except Exception as e:
            print(f"Could not create visualization: {e}")
    
    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
