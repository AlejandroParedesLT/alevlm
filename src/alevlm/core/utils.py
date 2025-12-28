import torch
import torch.nn as nn
import torch.optim as optim
from typing import List
import numpy as np
from numpy.typing import NDArray
import os
from typing import *
import typing
import json
from torch.utils.tensorboard import SummaryWriter
import os
import time

from alevlm.core.optim import (
    AdamW,
    gradient_cliping,
    CosineAnnealingWarmup
)
from alevlm.core.modules import (
    CELoss,
    neural_operation
)
from alevlm.core.inference import (
    DynamicCache
)

from PIL import Image
import webdataset as wds
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import json
import io

def perplexity(loss:list[torch.tensor]):
    m=len(loss)
    return torch.exp(1/m*torch.sum(loss))

def preprocess(sample,max_caption_length):
    """Process each sample from WebDataset"""
    # Define transform pipeline
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    # Decode image from JPEG bytes
    image = Image.open(io.BytesIO(sample['jpg'])).convert('RGB')
    image = image_transform(image)
    
    # Parse metadata
    metadata = json.loads(sample['json'])
    tokens = metadata['tokens']
    
    # Pad or truncate to max_caption_length
    if len(tokens) > max_caption_length:
        tokens = tokens[:max_caption_length]
    else:
        tokens = tokens + [0] * (max_caption_length - len(tokens))
    
    tokens = torch.tensor(tokens, dtype=torch.long)
    
    return {
        'image': image,
        'tokens': tokens,
        'text': sample['txt']
    }

def create_webdataset_loader(
    data_dir: str,
    split: str,
    batch_size: int,
    img_size: int = 224,
    max_caption_length: int = 77,
    num_workers: int = 4,
    shuffle_buffer: int = 1000
):
    """
    Create WebDataset dataloader with infinite iteration support
    """
    
    
    
    
    # Create dataset from shards
    urls = f"{data_dir}/{split}-*.tar"
    
    dataset = (
        wds.Dataset(urls)
        .shuffle(shuffle_buffer)
        .decode("rgb")
        .map(preprocess)
        .batched(batch_size, partial=False)
    )
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None, 
        num_workers=num_workers,
        pin_memory=True
    ) 
    
    return loader

def data_loading(
    x: NDArray[np.int_],
    batch_size,
    context_length,
    device='cuda:0'
    ):
    # Check if dataset is large enough for at least one sequence
    if x.shape[0] < context_length + 1:
        raise ValueError(
            f'Dataset too small: need at least {context_length + 1} tokens, '
            f'got {x.shape[0]}'
        )
    
    # Sample random starting indices for each sequence in the batch
    # We need context_length + 1 tokens (for input and target)
    # So valid starting positions are 0 to len(x) - context_length - 1
    num_possible_starts = x.shape[0] - context_length
    start_indices = np.random.randint(0, num_possible_starts, size=batch_size)
    
    # Extract sequences
    inputs = np.stack([x[i:i + context_length] for i in start_indices])
    targets = np.stack([x[i + 1:i + context_length + 1] for i in start_indices])
    
    # Convert to tensors and move to device
    inputs = torch.from_numpy(inputs).to(device)
    targets = torch.from_numpy(targets).to(device)
    
    return inputs, targets

# tstsh ?'[ ]

def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, iteration: int, out: str):
    model_obj = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iteration': iteration
    }
    print('saving to:',out)
    torch.save(model_obj, out)
    
    
    
def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: nn.Module,
    optimizer: optim.Optimizer = None,
    device: str | torch.device = None,
    strict: bool = True
):
    """
    Load model checkpoint and restore training state.
    
    Args:
        src: Path to checkpoint file or file-like object
        model: PyTorch model to load state into
        optimizer: Optional PyTorch optimizer to load state into
        device: Device to load checkpoint on (e.g., 'cuda', 'cpu')
        strict: Whether to strictly enforce state_dict keys match
    
    Returns:
        Tuple of (model, optimizer, iteration) if optimizer provided
        Tuple of (model, iteration) if optimizer is None
    """
    # Determine device
    if device is None:
        device = next(model.parameters()).device
    
    # Load checkpoint
    if isinstance(src, (str, os.PathLike)):
        # Load from file path
        chkpt = torch.load(src, map_location=device)
    elif hasattr(src, 'read'):
        # Load from file-like object
        chkpt = torch.load(src, map_location=device)
    else:
        raise TypeError(f"Unsupported input type: {type(src)}")
    
    # Load model state
    model.load_state_dict(chkpt['model'], strict=strict)
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer' in chkpt:
        optimizer.load_state_dict(chkpt['optimizer'])
    
    # Get iteration
    iteration = chkpt.get('iteration', 0)
    
    print(f"Checkpoint loaded from iteration {iteration}")
    
    if optimizer is not None:
        return model, optimizer, iteration
    else:
        return model, iteration

def training_together(
        model:nn.Module,
        train_path:str,
        valid_path:str,
        vocab_size: int,
        context_length: int,
        # Vision modules
        img_size:int,
        # Training parameters
        batch_size:int,
        lr: float,
        alpha_max: float,
        alpha_min: float,
        T_w: int,
        T_c: int,
        weight_decay: float,
        max_iters: int,
        log_interval: int,
        ckpt_path: str,
        betas=(0.9,0.999),
        eps_optimizer=1e-7,
        device: str = 'cuda',
        prefix_name_experiment:str='experiment',
    ):
    timestamp = int(time.time())
    writer = SummaryWriter(log_dir=f"runs/{prefix_name_experiment}_{timestamp}")
    

    # Create dataloaders (infinite iteration)
    print("Creating dataloaders...")
    train_loader = create_webdataset_loader(
        data_dir=train_path,
        split='train',
        batch_size=batch_size,
        img_size=img_size,
        max_caption_length=context_length,
        num_workers=4,
        shuffle_buffer=1000
    )
    
    valid_loader = create_webdataset_loader(
        data_dir=valid_path,
        split='validation',
        batch_size=batch_size,
        img_size=img_size,
        max_caption_length=context_length,
        num_workers=2,
        shuffle_buffer=100
    )
    
    train_iter = iter(train_loader)
    valid_iter = iter(valid_loader)

    model=model.to(device)
    optimizer=AdamW(params=model.parameters(),lr=lr,weight_decay=weight_decay,betas=betas,eps=eps_optimizer)
    criterion=CELoss()
    scheduler=CosineAnnealingWarmup(optimizer=optimizer,warmup_iters=T_c,max_iters=T_w,min_lr=alpha_min,max_lr=alpha_max)
    
    model.train()
    print(f"Starting training for {max_iters} iterations...")
    
    for iteration in range(1, max_iters + 1):
        # Get next batch from infinite iterator
        batch = next(train_iter)
        
        # Extract images and tokens
        images = batch['image'].to(device)  # [B, 3, H, W]
        tokens = batch['tokens'].to(device)  # [B, context_length]
        
        # Create input (all tokens except last) and target (all tokens except first)
        x = tokens[:, :-1]  # [B, context_length-1]
        y = tokens[:, 1:]   # [B, context_length-1]
        
        # Clear gradient accumulation
        optimizer.zero_grad()
        
        # Forward pass with images
        logits = model(x, img_x=images)  # Your VLM forward pass
        
        # Compute loss
        loss = criterion(
            logits.reshape(-1, vocab_size),
            y.reshape(-1)
        )
        
        # Backward pass
        loss.backward()
        gradient_cliping(model.parameters(), max_l2_norm=1.0, M=1.0)
        optimizer.step()
        scheduler.step()
        
        # Validation and logging
        if iteration % log_interval == 0:
            model.eval()
            with torch.no_grad():
                # Get validation batch
                val_batch = next(valid_iter)
                val_images = val_batch['image'].to(device)
                val_tokens = val_batch['tokens'].to(device)
                
                xv = val_tokens[:, :-1]
                yv = val_tokens[:, 1:]
                
                logits_v = model(xv, img_x=val_images)
                val_loss = criterion(
                    logits_v.reshape(-1, vocab_size),
                    yv.reshape(-1)
                ).item()
            
            print(
                f'iter {iteration}/{max_iters} | '
                f'train loss {loss.item():.4f} | '
                f'valid loss {val_loss:.4f} | '
                f'lr {scheduler.get_last_lr()[0]:.6f}'
            )
            
            # TensorBoard logging
            writer.add_scalar('Loss/train', loss.item(), iteration)
            writer.add_scalar('Loss/valid', val_loss, iteration)
            writer.add_scalar('LR', scheduler.get_last_lr()[0], iteration)
            
            model.train()
        
        # Save checkpoint
        if iteration % (10 * log_interval) == 0:
            print(f'Saving checkpoint to {ckpt_path}')
            save_checkpoint(model, optimizer, iteration, ckpt_path)
    
    # Final checkpoint
    save_checkpoint(model, optimizer, iteration, ckpt_path)
    writer.close()
    print("Training complete!")


def decoding(
        model:nn.Module, 
        x:torch.Tensor,
        max_num_tokens:int=1028,
        temperature:float=1.0,
        use_top_p:bool=False,
        use_top_k:bool=False,
        p:float=None,
        k:int=None,
        eos_token:int=256,
        use_cache:bool=False,
    ):
    """
    Autoregressive decoding with optional KV caching.
    
    Args:
        model: The language model
        x: Input token IDs [batch, seq_len]
        max_num_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        use_top_p: Whether to use top-p sampling
        use_top_k: Whether to use top-k sampling
        p: Top-p threshold (required if use_top_p=True)
        k: Top-k value (required if use_top_k=True)
        eos_token: End-of-sequence token ID
        use_cache: Whether to use KV caching for faster generation
    
    Returns:
        Generated token IDs [batch, seq_len + num_generated]
    """
    tokens_generated=0
    eos_flag=False
    original_shape=x.shape
    kv_cache= DynamicCache() if use_cache else None
    
    while tokens_generated<max_num_tokens and not eos_flag:
        with torch.no_grad():
            if use_cache:
                if kv_cache.layers and len(kv_cache.layers) > 0:
                    # Get number of cached tokens for the first layer (assume all layers synced)
                    seq_offset = kv_cache.layers[0].get_seq_length()
                else:
                    seq_offset = 0

                x_step = x if seq_offset == 0 else x[:, -1:]
                # Generate positions relative to cache
                # Get the length of the sequence
                cache_position = torch.arange(x_step.shape[1], device=x.device) + seq_offset

                logits=model(x_step,past_kv_caches=kv_cache,cache_position=cache_position)
            else:
                logits=model(x)
            logits=logits[:,-1,:]
            if use_top_p: 
                if not p:
                    raise ValueError('Need p parameter')
                next_token=neural_operation.top_p_sampling(logits=logits,p=p,temperature=temperature)
            elif use_top_k:
                if not k:
                    raise ValueError('Need k parameter')
                next_token=neural_operation.top_k_sampling(logits=logits,k=k,temperature=temperature)
            else:
                scaled_logits=logits/temperature
                probs=neural_operation.softmax(scaled_logits,dim=-1)
                next_token=torch.multinomial(probs,num_samples=1)
            
            x=torch.concat([x,next_token],dim=-1)
            tokens_generated+=1
            if (next_token == eos_token).any():
                eos_flag = True
        
    if x.shape==original_shape:
        print('The language model was not able to generate text')
    return x