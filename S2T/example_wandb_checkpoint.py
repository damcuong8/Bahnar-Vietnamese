"""
Example: How to Download and Use Wandb Checkpoint Artifacts

This script demonstrates how to download checkpoints from wandb 
and resume training or use for inference.
"""

import wandb
import torch
from speech2text_model import SeamlessM4Tv2ForSpeechToTextTrain_Pivot
from seamless_m4t_v2_config import SeamlessM4Tv2Config

# ============================================================================
# Example 1: Download Latest Checkpoint
# ============================================================================

def download_latest_checkpoint(project_name, artifact_name="model-checkpoint-step-5000-stage-A"):
    """
    Download the latest version of a checkpoint artifact
    
    Args:
        project_name: Your wandb project name
        artifact_name: Name of the artifact (without version)
        
    Returns:
        Path to downloaded checkpoint
    """
    # Initialize wandb (can be done without starting a new run)
    api = wandb.Api()
    
    # Get the artifact
    artifact = api.artifact(f'{project_name}/{artifact_name}:latest')
    
    # Download to local directory
    artifact_dir = artifact.download()
    
    checkpoint_path = f"{artifact_dir}/pytorch_model.bin"
    print(f"✓ Downloaded checkpoint to: {checkpoint_path}")
    
    return checkpoint_path


# ============================================================================
# Example 2: Load Checkpoint and Resume Training
# ============================================================================

def load_checkpoint_for_training(checkpoint_path):
    """
    Load checkpoint and prepare for resuming training
    
    Args:
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        model, optimizer, scheduler, metadata
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract metadata
    step = checkpoint['step']
    stage = checkpoint.get('stage', 'unknown')
    epoch = checkpoint['epoch']
    
    print(f"Checkpoint info:")
    print(f"  Step: {step}")
    print(f"  Stage: {stage}")
    print(f"  Epoch: {epoch}")
    
    # Create model
    config = SeamlessM4Tv2Config()
    model = SeamlessM4Tv2ForSpeechToTextTrain_Pivot(config)
    
    # Load model weights
    model.load_state_dict(checkpoint['model'])
    print("✓ Model weights loaded")
    
    # Create optimizer and load state
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("✓ Optimizer state loaded")
    
    # Load scheduler state if available
    scheduler = None
    if 'scheduler' in checkpoint:
        from torch.optim.lr_scheduler import LambdaLR
        scheduler = LambdaLR(optimizer, lambda x: 1.0)  # Dummy
        scheduler.load_state_dict(checkpoint['scheduler'])
        print("✓ Scheduler state loaded")
    
    metadata = {
        'step': step,
        'stage': stage,
        'epoch': epoch,
    }
    
    return model, optimizer, scheduler, metadata


# ============================================================================
# Example 3: Compare Multiple Checkpoints
# ============================================================================

def compare_checkpoints(project_name, artifact_names):
    """
    Download and compare metadata from multiple checkpoints
    
    Args:
        project_name: Your wandb project name
        artifact_names: List of artifact names to compare
    """
    api = wandb.Api()
    
    print("Comparing checkpoints:\n")
    print(f"{'Artifact':<50} {'Step':<10} {'Stage':<10}")
    print("-" * 70)
    
    for artifact_name in artifact_names:
        try:
            artifact = api.artifact(f'{project_name}/{artifact_name}:latest')
            metadata = artifact.metadata
            
            step = metadata.get('step', 'N/A')
            stage = metadata.get('stage', 'N/A')
            
            print(f"{artifact_name:<50} {step:<10} {stage:<10}")
        except Exception as e:
            print(f"{artifact_name:<50} ERROR: {e}")


# ============================================================================
# Example 4: Download Best Checkpoint Based on Metrics
# ============================================================================

def find_best_checkpoint(project_name, metric='train/loss', mode='min'):
    """
    Find and download the checkpoint with the best metric value
    
    Args:
        project_name: Your wandb project name
        metric: Metric to optimize (e.g., 'train/loss', 'train/ce_loss')
        mode: 'min' or 'max'
        
    Returns:
        Path to best checkpoint
    """
    api = wandb.Api()
    
    # Get all runs in the project
    runs = api.runs(project_name)
    
    best_value = float('inf') if mode == 'min' else float('-inf')
    best_artifact = None
    
    for run in runs:
        # Get run summary (final metrics)
        summary = run.summary
        
        if metric in summary:
            value = summary[metric]
            
            # Check if this is better
            is_better = (value < best_value) if mode == 'min' else (value > best_value)
            
            if is_better:
                best_value = value
                
                # Get artifacts from this run
                artifacts = run.logged_artifacts()
                for artifact in artifacts:
                    if artifact.type == 'model':
                        best_artifact = artifact
                        break
    
    if best_artifact:
        print(f"✓ Found best checkpoint: {best_artifact.name}")
        print(f"  Metric {metric}: {best_value}")
        
        artifact_dir = best_artifact.download()
        checkpoint_path = f"{artifact_dir}/pytorch_model.bin"
        return checkpoint_path
    else:
        print(f"✗ No checkpoint found with metric {metric}")
        return None


# ============================================================================
# Example 5: List All Available Checkpoints
# ============================================================================

def list_all_checkpoints(project_name):
    """
    List all checkpoint artifacts in a project
    
    Args:
        project_name: Your wandb project name
    """
    api = wandb.Api()
    
    # Get all artifacts of type 'model'
    artifacts = api.artifacts(type_name='model', project=project_name)
    
    print(f"Available checkpoints in {project_name}:\n")
    
    for artifact in artifacts:
        print(f"Name: {artifact.name}")
        print(f"  Created: {artifact.created_at}")
        print(f"  Size: {artifact.size / (1024**3):.2f} GB")
        
        if hasattr(artifact, 'metadata') and artifact.metadata:
            print(f"  Metadata:")
            for key, value in artifact.metadata.items():
                print(f"    {key}: {value}")
        print()


# ============================================================================
# Example 6: Use Checkpoint for Inference
# ============================================================================

def load_for_inference(checkpoint_path, device='cuda'):
    """
    Load checkpoint for inference only
    
    Args:
        checkpoint_path: Path to checkpoint
        device: Device to load model on
        
    Returns:
        model ready for inference
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model
    config = SeamlessM4Tv2Config()
    model = SeamlessM4Tv2ForSpeechToTextTrain_Pivot(config)
    
    # Load weights
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()  # Set to evaluation mode
    
    print(f"✓ Model loaded for inference on {device}")
    print(f"  From step: {checkpoint['step']}")
    print(f"  Stage: {checkpoint.get('stage', 'unknown')}")
    
    return model


# ============================================================================
# Main Examples
# ============================================================================

if __name__ == "__main__":
    PROJECT_NAME = "seamlessm4t-v2-finetuning-S2T"
    
    print("="*70)
    print("Wandb Checkpoint Examples")
    print("="*70 + "\n")
    
    # Example 1: Download latest
    print("\n1. Download Latest Checkpoint")
    print("-" * 70)
    try:
        checkpoint_path = download_latest_checkpoint(
            PROJECT_NAME,
            "model-checkpoint-step-5000-stage-A"
        )
    except Exception as e:
        print(f"Note: {e}")
        print("Make sure you have uploaded checkpoints first!")
    
    # Example 2: List all checkpoints
    print("\n2. List All Checkpoints")
    print("-" * 70)
    try:
        list_all_checkpoints(PROJECT_NAME)
    except Exception as e:
        print(f"Note: {e}")
    
    # Example 3: Compare checkpoints
    print("\n3. Compare Multiple Checkpoints")
    print("-" * 70)
    try:
        compare_checkpoints(PROJECT_NAME, [
            "model-checkpoint-step-5000-stage-A",
            "model-checkpoint-step-8000-stage-B",
            "model-checkpoint-step-15000-stage-C",
        ])
    except Exception as e:
        print(f"Note: {e}")
    
    # Example 4: Find best checkpoint
    print("\n4. Find Best Checkpoint by Loss")
    print("-" * 70)
    try:
        best_ckpt = find_best_checkpoint(PROJECT_NAME, metric='train/loss', mode='min')
        if best_ckpt:
            print(f"Best checkpoint path: {best_ckpt}")
    except Exception as e:
        print(f"Note: {e}")
    
    print("\n" + "="*70)
    print("Examples complete!")
    print("="*70)
    
    print("\nTo use in your code:")
    print("""
    # Simple download and load
    import wandb
    
    api = wandb.Api()
    artifact = api.artifact('your-project/model-checkpoint-step-8000-stage-B:latest')
    artifact_dir = artifact.download()
    
    checkpoint = torch.load(f"{artifact_dir}/pytorch_model.bin")
    model.load_state_dict(checkpoint['model'])
    """)

