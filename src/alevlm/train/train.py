import torch
# import torchvision
import torch.nn as nn
import torch.optim as optim
from typing import List
import numpy as np
from numpy.typing import NDArray
import os
from typing import *
import typing
import json

from alevlm.core.modules import (
    MoEVLM
)
from alevlm.core.utils import (
    training_together
)
import argparse
import json
import yaml
import os
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, List

@dataclass
class TrainingConfig:
    """Training configuration - flexible to accept any config structure"""
    
    # Model architecture
    vocab_size: int = 50257
    context_length: int = 256
    d_model: int = 512
    d_ff: int = 1344
    rope_theta: float = 10000.0
    n_layers: int = 4
    n_heads: int = 16

    # Image settings
    num_experts: int = 16
    img_size: int=256
    patch_size: int = 2
    use_images: bool = True
    image_d_model:int = 192
    top_k:int = 30,
    expert_d: int= 512

    # Training hyperparameters
    lr: float = 0.001
    lr_warmup: float = 0.001
    alpha_max:float = 6e-4    # Peak learning rate
    alpha_min:float = 6e-6    # Minimum learning rate (alpha_max / 10)
    T_w:int = 1000          # Warmup iterations (10% of total)
    T_c:int = 15000         # Total iterations
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.001
    
    # Data parameters
    train_data: str = "./data/train.txt"
    val_data: Optional[str] = None

    
    # Output and logging
    ckpt_path: str = "./outputs"
    log_interval: int = 100
    save_steps: int = 1000
    eval_steps: int = 500
    batch_size: int = 1028
    max_iters: int = 10000
    prefix_name_experiment: str = 'experiment'
    
    # Hardware
    device: str = "cuda"
    mixed_precision: bool = False
    dtype: str = "float64"
    optimized: bool=False
    
    # Other
    seed: int = 42
    debug: bool = False


def load_config(config_path: str) -> TrainingConfig:
    """Load configuration from YAML or JSON file"""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    print(f"Loading config from: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.suffix in ['.yaml', '.yml']:
            config_dict = yaml.safe_load(f)
        elif config_path.suffix == '.json':
            config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    # Create config object with loaded values
    return TrainingConfig(**config_dict)


def override_config(config: TrainingConfig, overrides: dict) -> TrainingConfig:
    """Override config values with CLI arguments"""
    config_dict = asdict(config)
    
    # Only override non-None values
    for key, value in overrides.items():
        if value is not None and key in config_dict:
            config_dict[key] = value
            print(f"  Overriding {key}: {config_dict[key]}")
    
    return TrainingConfig(**config_dict)


def parse_args():
    """Parse command-line arguments for config overrides"""
    parser = argparse.ArgumentParser(
        description='LLM Training with Config File Management',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Config file (required in production)
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file (YAML or JSON) - REQUIRED')
    
    # Common overrides for experiments
    parser.add_argument('--experiment_name', type=str,
                        help='Override experiment name')
    parser.add_argument('--batch_size', type=int,
                        help='Override batch size')
    parser.add_argument('--learning_rate', type=float,
                        help='Override learning rate')
    parser.add_argument('--num_epochs', type=int,
                        help='Override number of epochs')
    parser.add_argument('--output_dir', type=str,
                        help='Override output directory')
    
    # Quick toggles
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    parser.add_argument('--eval_only', action='store_true',
                        help='Run evaluation only')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Enable mixed precision training')
    
    # Resume training
    parser.add_argument('--resume_from', type=str,
                        help='Resume from checkpoint')
    
    # Device override
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu', 'mps'],
                        help='Override device')
    
    return parser.parse_args()

def save_config(config: TrainingConfig, output_dir: str, filename: str = 'config.yaml'):
    """Save configuration to output directory"""
    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(asdict(config), f, default_flow_style=False, sort_keys=False)
    
    print(f"✓ Configuration saved to: {output_path}")


def print_config(config: TrainingConfig):
    """Pretty print configuration"""
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    
    config_dict = asdict(config)
    sections = {
        "Model": ["model_name", "model_size", "pretrained_path"],
        "Training": ["batch_size", "learning_rate", "num_epochs", "max_steps", 
                    "warmup_steps", "gradient_accumulation_steps"],
        "Data": ["train_data", "val_data", "test_data", "max_seq_length"],
        "Optimization": ["optimizer", "weight_decay", "grad_clip", "lr_scheduler"],
        "Output": ["output_dir", "experiment_name", "save_steps", "eval_steps"],
        "Hardware": ["device", "num_workers", "mixed_precision", "distributed"],
        "Other": ["seed", "debug", "eval_only", "resume_from"],
    }
    
    for section, keys in sections.items():
        print(f"\n{section}:")
        for key in keys:
            if key in config_dict:
                value = config_dict[key]
                if value is not None and value != "" and value != []:
                    print(f"  {key:30s}: {value}")
    
    print("\n" + "="*60 + "\n")


def main():
    """Main function - config-file first approach"""
    args = parse_args()
    
    # Load base configuration from file
    config = load_config(args.config)
    
    # Apply CLI overrides if provided
    overrides = {k: v for k, v in vars(args).items() if k != 'config'}
    if any(v is not None for v in overrides.values()):
        print("\nApplying CLI overrides:")
        config = override_config(config, overrides)
    
    # Validate configuration
    # validate_config(config)
    
    dtype_args=None
    if config.dtype=='float64':
        dtype_args=torch.float64
    elif config.dtype=='float32':
        dtype_args=torch.float32
    else:
        dtype_args=torch.float16

    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent
    train_data_dir = project_root / "data" #/ config.train_data #"TinyStoriesV2-GPT4-train.txt"
    valid_data_dir = project_root / "data" #/ config.val_data #"TinyStoriesV2-GPT4-valid.txt"
    ckpt_path = project_root / "assets" / config.ckpt_path

    alevlm=MoEVLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_ff=config.d_ff,
        d_model=config.d_model,
        top_k=config.top_k,
        num_heads=config.n_heads,
        expert_d=config.expert_d,
        num_layers=config.n_layers,
        rope_theta=config.rope_theta,
        num_experts=config.num_experts,
        img_size=config.img_size,
        image_d_model=config.image_d_model,
        patch_size=config.patch_size,
        use_images=config.use_images,
        device=config.device,
        dtype=dtype_args
    )
    
    # Note:

    # Config for 6gb memory:
    # 6* Batch Size * Seq * d_model * 4

    def count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # def count_params_by_module(model):
    #     for name, p in model.named_parameters():
    #         if p.requires_grad:
    #             print(name, p.numel())

    total_params = count_params(alevlm)
    print(f"Total parameters: {total_params:,}")

    training_together(
        model=alevlm,
        train_path=train_data_dir,
        valid_path=valid_data_dir,
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        # Image parameters
        img_size=config.img_size, 
        # Others
        batch_size=config.batch_size,
        lr=config.lr,
        alpha_max=config.alpha_max,  
        alpha_min=config.alpha_min,  
        T_w=config.T_w,
        T_c=config.T_c,  
        weight_decay=config.weight_decay,
        max_iters=config.max_iters,
        log_interval=config.log_interval,
        ckpt_path=ckpt_path,
        betas=(config.beta1,config.beta2),
        eps_optimizer=config.eps,
        device=config.device,
        prefix_name_experiment=config.prefix_name_experiment
    )
    print('Finished training!')
    

if __name__=='__main__':
    main()