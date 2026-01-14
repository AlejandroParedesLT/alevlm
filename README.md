# AleMoEVLM 🚀

From-scratch implementation of a Mixture-of-Experts Vision-Language Model with modern transformer architecture, featuring sparse expert routing, rotary positional embeddings, and vision-language fusion capabilities.

## 🌟 Features

### Core Architecture
- **Sparse Mixture-of-Experts (MoE)**: Efficient parameter scaling with top-k expert routing
- **Vision Transformer (ViT)**: Custom patch embedding and vision encoding
- **Rotary Position Embeddings (RoPE)**: Position-aware attention without learned embeddings
- **Multi-Modal Fusion**: Seamless integration of vision and language modalities
- **Causal Language Modeling**: Autoregressive text generation with KV caching

### Implementation Highlights
- ✨ **Built from scratch** - No high-level transformer libraries, pure PyTorch implementation
- 🔧 **Modular design** - Easy to extend and customize components
- ⚡ **Optimized inference** - Dynamic KV caching for efficient generation
- 🎯 **Flexible training** - YAML/JSON configuration system with CLI overrides
- 📊 **Load balancing** - Auxiliary loss to ensure uniform expert utilization

## 📋 Table of Contents

- [Architecture Overview](#-architecture-overview)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Model Components](#-model-components)
- [Training](#-training)
- [Configuration](#-configuration)
- [Advanced Usage](#-advanced-usage)
- [Performance](#-performance)
- [Citation](#-citation)

## 🏗️ Architecture Overview

```
AleMoEVLM Architecture
│
├── Vision Encoder (ViT)
│   ├── Patch Embeddings (Conv2D)
│   ├── CLS Token
│   ├── Transformer Blocks (RoPE + Self-Attention)
│   └── Layer Normalization (RMSNorm)
│
├── Language Model
│   ├── Token Embeddings
│   ├── Vision Projection (optional)
│   ├── Sparse MoE Blocks
│   │   ├── Multi-Head Self-Attention (RoPE)
│   │   ├── Noisy Top-k Router
│   │   └── Expert FFN Layers
│   └── Language Modeling Head
│
└── Auxiliary Components
    ├── Dynamic KV Cache
    ├── Custom Loss Functions
    └── Sampling Strategies (Top-p, Top-k)
```

### Key Innovations

**1. Sparse MoE with Noisy Routing**
- Top-k expert selection per token
- Gaussian noise during training for exploration
- Load balancing loss to prevent expert collapse

**2. Rotary Position Embeddings**
- Relative position encoding without learned parameters
- Works seamlessly with variable sequence lengths
- Supports both 3D and 4D tensor inputs

**3. Vision-Language Fusion**
- Flexible image integration via projection layer
- Compatible with variable image sizes
- Optional vision encoding for text-only training

## 📦 Installation

### Requirements
```bash
Python >= 3.8
PyTorch >= 2.0
einops >= 0.6.0
numpy >= 1.21.0
PyYAML >= 6.0
```

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/alemoe-vlm.git
cd alemoe-vlm

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Project Structure
```
alemoe-vlm/
├── alevlm/
│   ├── core/
│   │   ├── modules.py          # Model architecture
│   │   ├── inference.py        # KV cache & inference utilities
│   │   └── utils.py            # Training utilities
│   └── scripts/
│       └── train.py            # Training script
├── configs/
│   ├── base_config.yaml        # Base configuration
│   └── experiments/            # Experiment configs
├── data/                       # Training data
├── assets/                     # Model checkpoints
└── README.md
```

## 🚀 Quick Start

### Basic Training

```python
from alevlm.core.modules import MoEVLM
import torch

# Initialize model
model = AleMoEVLM(
    vocab_size=50257,
    context_length=256,
    d_model=512,
    num_heads=16,
    num_layers=4,
    num_experts=16,
    top_k=4,
    expert_d=512,
    d_ff=1344,
    rope_theta=10000.0,
    image_d_model=192,
    img_size=256,
    patch_size=16,
    use_images=True,
    device='cuda',
    dtype=torch.float32
)

# Forward pass (text only)
input_ids = torch.randint(0, 50257, (4, 128))  # [batch, seq_len]
logits = model(input_ids)  # [batch, seq_len, vocab_size]

# Forward pass (with images)
images = torch.randn(4, 3, 256, 256)  # [batch, channels, height, width]
logits = model(input_ids, img_x=images)
```

### Training with Config

```bash
# Using a config file
python -m alevlm.scripts.train --config configs/base_config.yaml

# Override specific parameters
python -m alevlm.scripts.train \
    --config configs/base_config.yaml \
    --batch_size 64 \
    --learning_rate 0.001 \
    --experiment_name my_experiment

# Debug mode
python -m alevlm.scripts.train \
    --config configs/base_config.yaml \
    --debug \
    --device cuda
```

## 🧩 Model Components

### 1. Multi-Head Self-Attention with RoPE

```python
from alevlm.core.modules import MultiheadSelfAttention

attn = MultiheadSelfAttention(
    d_model=512,
    num_heads=8,
    rope_theta=10000,
    max_seq_len=2048,
    layer_idx=0,
    is_decoder=True
)

# With KV caching for inference
output = attn(x, past_kv_values=cache, cache_position=positions)
```

### 2. Sparse Mixture-of-Experts

```python
from alevlm.core.modules import SparseMoE

moe = SparseMoE(
    d_model=512,
    num_experts=16,
    top_k=4,
    expert_d=2048
)

output, aux_loss = moe(x)  # aux_loss for load balancing
```

### 3. Vision Transformer

```python
from alevlm.core.modules import VIT

vit = VIT(
    img_size=224,
    patch_size=16,
    d_model=768,
    num_heads=12,
    rope_theta=10000,
    n_blocks=12
)

image_embeddings = vit(images)  # [batch, d_model]
```

### 4. Custom Sampling

```python
from alevlm.core.modules import neural_operation

# Top-p (nucleus) sampling
next_token = neural_operation.top_p_sampling(
    logits, 
    p=0.9, 
    temperature=0.8
)

# Top-k sampling
next_token = neural_operation.top_k_sampling(
    logits, 
    k=50, 
    temperature=1.0
)
```

## 🎓 Training

### Configuration System

AleMoEVLM uses a flexible YAML/JSON configuration system:

```yaml
# config.yaml
# Model Architecture
vocab_size: 50257
context_length: 256
d_model: 512
n_heads: 16
n_layers: 4
d_ff: 1344

# MoE Settings
num_experts: 16
top_k: 4
expert_d: 512

# Vision Settings
use_images: true
img_size: 256
patch_size: 16
image_d_model: 192

# Training Hyperparameters
batch_size: 64
alpha_max: 6e-4      # Peak learning rate
alpha_min: 6e-6      # Min learning rate
T_w: 1000           # Warmup steps
T_c: 15000          # Total steps
weight_decay: 0.001

# Optimization
beta1: 0.9
beta2: 0.999
eps: 1e-8

# Data
train_data: "./data/train.txt"
val_data: "./data/val.txt"

# Output
ckpt_path: "./outputs"
log_interval: 100
save_steps: 1000
eval_steps: 500
prefix_name_experiment: "alemoe_baseline"

# Hardware
device: "cuda"
dtype: "float32"
mixed_precision: false
```

### Training Pipeline

The training process includes:
- ✅ Cosine learning rate schedule with warmup
- ✅ Gradient clipping and weight decay
- ✅ Periodic evaluation and checkpointing
- ✅ Load balancing loss for MoE
- ✅ Mixed precision support (optional)

### Monitoring Training

```bash
# Training will log:
# - Training loss
# - Validation loss (if val_data provided)
# - Expert utilization statistics
# - Learning rate schedule
# - Checkpoint saves
```

## ⚙️ Configuration

### Complete Configuration Options

| Category | Parameter | Default | Description |
|----------|-----------|---------|-------------|
| **Model** | `vocab_size` | 50257 | Vocabulary size |
| | `context_length` | 256 | Maximum sequence length |
| | `d_model` | 512 | Model dimension |
| | `n_heads` | 16 | Number of attention heads |
| | `n_layers` | 4 | Number of transformer layers |
| | `d_ff` | 1344 | Feed-forward dimension |
| | `rope_theta` | 10000.0 | RoPE base frequency |
| **MoE** | `num_experts` | 16 | Total number of experts |
| | `top_k` | 4 | Experts selected per token |
| | `expert_d` | 512 | Expert hidden dimension |
| **Vision** | `use_images` | true | Enable vision encoder |
| | `img_size` | 256 | Input image size |
| | `patch_size` | 16 | Vision patch size |
| | `image_d_model` | 192 | Vision encoder dimension |
| **Training** | `batch_size` | 64 | Training batch size |
| | `alpha_max` | 6e-4 | Peak learning rate |
| | `alpha_min` | 6e-6 | Minimum learning rate |
| | `T_w` | 1000 | Warmup iterations |
| | `T_c` | 15000 | Total iterations |
| | `weight_decay` | 0.001 | AdamW weight decay |

## 🔬 Advanced Usage

### Custom Expert Architecture

```python
from alevlm.core.modules import MoEExpert

class CustomExpert(nn.Module):
    def __init__(self, d_model, h_d, device, dtype):
        super().__init__()
        self.net = nn.Sequential(
            Linear(d_model, h_d, device=device, dtype=dtype),
            SiLU(),
            Linear(h_d, d_model, device=device, dtype=dtype)
        )
    
    def forward(self, x):
        return self.net(x)
```

### Inference with KV Caching

```python
from alevlm.core.inference import DynamicCache

cache = DynamicCache()
generated_tokens = []

for _ in range(max_new_tokens):
    logits = model(
        input_ids, 
        past_kv_values=cache,
        cache_position=positions
    )
    next_token = neural_operation.top_p_sampling(logits[:, -1, :])
    generated_tokens.append(next_token)
    input_ids = next_token
```

### Multi-Modal Training

```python
# Prepare vision-language pairs
for batch in dataloader:
    images, text_ids, targets = batch
    
    # Forward pass with vision
    logits = model(text_ids, img_x=images)
    
    # Compute loss
    loss = criterion(logits, targets)
    loss.backward()
    optimizer.step()
```

## 📊 Performance

### Memory Efficiency

The model's memory footprint can be estimated as:
```
Memory ≈ 6 × batch_size × seq_length × d_model × 4 bytes
```

For the default configuration:
- **Parameters**: ~100M (varies with configuration)
- **Memory (batch=32, seq=256)**: ~3GB GPU memory
- **Inference Speed**: ~50 tokens/sec (RTX 3090)

### Scaling Guidelines

| Model Size | d_model | n_layers | n_heads | num_experts | Parameters |
|------------|---------|----------|---------|-------------|------------|
| Small | 256 | 4 | 8 | 8 | ~25M |
| Base | 512 | 4 | 16 | 16 | ~100M |
| Large | 768 | 8 | 16 | 32 | ~300M |
| XL | 1024 | 12 | 16 | 64 | ~800M |

## 🛠️ Custom Components

### Implementing New Activation Functions

```python
class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)
        ))
```

### Custom Normalization

```python
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
```

## 🐛 Troubleshooting

### Common Issues

**Out of Memory**
```bash
# Reduce batch size
--batch_size 16

# Use gradient accumulation
--gradient_accumulation_steps 4

# Enable mixed precision
--mixed_precision
```

**Slow Training**
```bash
# Reduce number of experts
num_experts: 8
top_k: 2

# Decrease model size
d_model: 256
n_layers: 2
```

**Expert Collapse**
- Increase load balancing loss weight
- Adjust noise scale in router
- Use more training data

## 📚 References

This implementation draws inspiration from:
- **Sparse Mixture-of-Experts**: [Switch Transformers (Google, 2021)](https://arxiv.org/abs/2101.03961)
- **Rotary Embeddings**: [RoFormer (Su et al., 2021)](https://arxiv.org/abs/2104.09864)
- **Vision Transformers**: [ViT (Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929)
- **SwiGLU**: [GLU Variants (Shazeer, 2020)](https://arxiv.org/abs/2002.05202)

## 📄 License

MIT License - feel free to use this code for research and commercial applications.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

*AleMoEVLM - Where Vision Meets Language Through Sparse Expertise*