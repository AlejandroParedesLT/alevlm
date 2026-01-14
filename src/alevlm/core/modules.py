from torch import nn
import torch
from typing import Any, Optional
from einops import rearrange, einsum
from alevlm.core.inference import DynamicCache,DynamicLayer
from collections.abc import Callable

class Linear(nn.Module):
    def __init__(self, in_features:int,out_features:int,device:torch.device=None,dtype:torch.dtype=None, bias:bool=True):
        super().__init__()
        self.in_features=in_features
        self.out_features=out_features
        self.device=device
        self.dtype=dtype
        # Linear layer
        # self.layer=nn.Linear(in_features=self.in_features,out_features=self.out_features,bias=True, device=self.device,dtype=self.dtype)
        weights=torch.empty(self.out_features,self.in_features,device=self.device,dtype=self.dtype)
        # nn.init.trunc_normal_(weights)
        nn.init.kaiming_uniform_(weights, a=0, mode='fan_in', nonlinearity='linear')
        self.weight=nn.Parameter(weights)

        # bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, device=self.device, dtype=self.dtype))
        else:
            self.bias = None
        
    def forward(self, x:torch.Tensor) -> torch.Tensor: # dim [batch_size, sequence_length]
        y=x @ self.weight.T
        if self.bias is not None:
            y = y + self.bias
        return y


class Embedding(nn.Module):
    def __init__(self, num_embeddings:int, embedding_dim:int, device:torch.device=None, dtype:torch.dtype=None):
        super().__init__()
        self.num_embeddings=num_embeddings
        self.embedding_dim=embedding_dim
        self.device=device
        self.dtype=dtype
        weights=torch.empty(self.num_embeddings,self.embedding_dim,device=self.device,dtype=self.dtype)
        nn.init.normal_(weights, mean=0.0, std=1.0 / math.sqrt(self.embedding_dim))
        self.weight=nn.Parameter(weights)
    
    def forward(self, token_ids:torch.Tensor):
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model:int,eps:float=1e-5,device:torch.device=None,dtype:torch.dtype=None):
        super().__init__()
        self.d_model=d_model
        self.eps=eps
        self.device=device
        self.dtype=dtype
        w=torch.ones(self.d_model,device=self.device,dtype=self.dtype)
        self.gain=nn.Parameter(w)
        
    def forward(self, x:torch.Tensor)->torch.Tensor:
        x=x.to(torch.float32)
        rms_a=torch.sqrt((1/self.d_model)*torch.sum(x**2,dim=-1, keepdim=True)+self.eps)
        result=(x/rms_a)*self.gain
        return result.to(self.dtype)
    
class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
        
    
class swiGLU(nn.Module):
    def __init__(self,d_model:int,d_ff: int,device:torch.device=None,dtype:torch.dtype=None):
        super().__init__()
        self.d_model=d_model
        self.device=device
        self.dtype=dtype
        self.d_ff=d_ff#int((8/3)*self.d_model)
        self.w1=Linear(self.d_model,self.d_ff,device=self.device,dtype=self.dtype,bias=False)
        self.w2=Linear(self.d_model,self.d_ff,device=self.device,dtype=self.dtype,bias=False)
        self.w3=Linear(self.d_ff,self.d_model,device=self.device,dtype=self.dtype,bias=False)
        self.silu=SiLU()

    def forward(self, x:torch.Tensor):
        siluProj=self.w1(x)
        silu=self.silu(siluProj)
        gate=self.w2(x)
        glu=silu*gate
        swiglu=self.w3(glu)
        return swiglu
    
class RoPE(nn.Module):
    def __init__(self, theta:float, d_k:int, max_seq_len:int, device:torch.device=None,dtype:torch.dtype=None):
        super().__init__()
        self.theta=theta
        self.d_k=d_k
        self.max_seq_len=max_seq_len
        self.device=device
        self.dtype = dtype if dtype is not None else torch.float32
        
        # Implementation
        position_indices=torch.arange(max_seq_len, device=device, dtype=dtype).unsqueeze(1)
        # Dimension indices for pairs: [0, 2, 4, ..., d_k-2]
        i = torch.arange(0, d_k, 2, device=device, dtype=dtype).float()   # [0, 2, 4, ..., d_k-2]
        # Compute frequencies: 1 / (theta^(2k/d)) for k in {0, 1, 2, ..., d_k/2-1}
        freqs = 1.0 / (theta ** (i / d_k))  # [d_k/2]
        
        angles = position_indices * freqs  # [max_seq_len, d_k/2]
        
        # Precompute cos and sin values
        cos_angles = torch.cos(angles).to(dtype=dtype)  # [max_seq_len, d_k/2]
        sin_angles = torch.sin(angles).to(dtype=dtype)  # [max_seq_len, d_k/2]
        
        self.register_buffer('cos_cached', cos_angles, persistent=False)
        self.register_buffer('sin_cached', sin_angles, persistent=False)
        
    def rotate_half(self,x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x:torch.Tensor, token_positions: torch.Tensor | None = None):
        """
        Apply rotary position embeddings to input tensor
        Args:
            x: tensor shape [batch size, seq len, d_k]
        Returns:
            Rotated tensor of same shape 
        """
        # Handle both 3D and 4D inputs
        if x.ndim == 4:
            batch_size, num_heads, seq_len, d_head = x.shape
            is_4d = True
        elif x.ndim == 3:
            batch_size, seq_len, d_head = x.shape
            num_heads = 1
            is_4d = False
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got {x.ndim}D")

        # if token_positions is None:
        #     # Get cos and sin for the seq length
        #     cos=self.cos_cached[:seq_len,:]
        #     sin=self.sin_cached[:seq_len,:]
        # else:
        #     cos = self.cos_cached[token_positions]
        #     sin = self.sin_cached[token_positions]
            
        # x_reshaped = rearrange(x, 'b s (d two) -> b s d two', two=2)
        
        #  # Extract the two elements of each pair
        # x1 = x_reshaped[..., 0]  # [batch_size, seq_len, d_k/2]
        # x2 = x_reshaped[..., 1]  # [batch_size, seq_len, d_k/2]
        
        # # Apply rotation matrix:
        # rotated_x1 = x1 * cos - x2 * sin  # [batch_size, seq_len, d_k/2]
        # rotated_x2 = x1 * sin + x2 * cos  # [batch_size, seq_len, d_k/2]
        
        # # Stack back into pairs and reshape to original shape
        # rotated = torch.stack([rotated_x1, rotated_x2], dim=-1)  # [batch_size, seq_len, d_k/2, 2]
        # rotated = rearrange(rotated, 'b s d two -> b s (d two)')  # [batch_size, seq_len, d_k]
        
        # return rotated
        if token_positions is None:
            # Get cos and sin for the seq length
            cos = self.cos_cached[:seq_len, :]  # [seq_len, d_head/2]
            sin = self.sin_cached[:seq_len, :]  # [seq_len, d_head/2]
        else:
            cos = self.cos_cached[token_positions]
            sin = self.sin_cached[token_positions]
        
        # Reshape based on input dimensions
        if is_4d:
            # x: [batch, heads, seq, d_head]
            # Rearrange to pairs
            x_reshaped = rearrange(x, 'b h s (d two) -> b h s d two', two=2)
            
            # Extract pairs
            x1 = x_reshaped[..., 0]  # [batch, heads, seq, d_head/2]
            x2 = x_reshaped[..., 1]  # [batch, heads, seq, d_head/2]
            
            # Expand cos/sin to match: [seq, d/2] -> [1, 1, seq, d/2]
            cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, d_head/2]
            sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, d_head/2]
            
            # Apply rotation
            rotated_x1 = x1 * cos - x2 * sin
            rotated_x2 = x1 * sin + x2 * cos
            
            # Stack and reshape back
            rotated = torch.stack([rotated_x1, rotated_x2], dim=-1)  # [batch, heads, seq, d/2, 2]
            rotated = rearrange(rotated, 'b h s d two -> b h s (d two)')  # [batch, heads, seq, d_head]
            
        else:  # 3D case
            # x: [batch, seq, d_head]
            x_reshaped = rearrange(x, 'b s (d two) -> b s d two', two=2)
            
            x1 = x_reshaped[..., 0]  # [batch, seq, d_head/2]
            x2 = x_reshaped[..., 1]  # [batch, seq, d_head/2]
            
            # Expand cos/sin: [seq, d/2] -> [1, seq, d/2]
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
            
            # Apply rotation
            rotated_x1 = x1 * cos - x2 * sin
            rotated_x2 = x1 * sin + x2 * cos
            
            # Stack and reshape back
            rotated = torch.stack([rotated_x1, rotated_x2], dim=-1)  # [batch, seq, d/2, 2]
            rotated = rearrange(rotated, 'b s d two -> b s (d two)')  # [batch, seq, d_head]
        
        return rotated


import math
class neural_operation:
    @staticmethod
    def softmax( x: torch.Tensor, dim: int):
        x_max = x.max(dim=dim, keepdim=True).values
        e_x = torch.exp(x - x_max)
        return e_x / e_x.sum(dim=dim, keepdim=True)
        
    @staticmethod
    def scaled_dot_product_attention(proj_q,proj_k,proj_v,mask):
        # (batch_size, ..., seq_len, d_k)
        # (batch_size, ..., seq_len, d_v),
        d_k = proj_q.size(-1)
        proj_k_t=proj_k.transpose(-2, -1) #rearrange(proj_k,'batch heads seq dim -> batch heads dim seq')
        scores = (proj_q @ proj_k_t) / torch.sqrt(torch.tensor(d_k, dtype=proj_q.dtype, device=proj_q.device))
        if mask is not None:
            mask = mask.to(scores.device)
            scores = scores.masked_fill(~mask, float('-inf'))
        attention_matrix=neural_operation.softmax(scores,dim=-1)
        output=attention_matrix @ proj_v
        return output
    
    @staticmethod
    def top_p_sampling(logits:torch.Tensor,p:float=0.9,temperature:float=1.0):
        """
        Perform top-p (nucleus) sampling on logits.
        
        Top-p sampling selects the smallest set of tokens whose cumulative probability
        exceeds the threshold p. This allows dynamic vocabulary size based on the
        probability distribution.
        
        Args:
            logits: Tensor of shape (batch_size, vocab_size) or (vocab_size,)
                    Raw logits from the model (unnormalized log probabilities)
            p: Float in (0, 1]. Cumulative probability threshold.
            p=1.0 means no filtering (sample from full distribution)
            p=0.9 means sample from top 90% probability mass
            temperature: Float > 0. Controls randomness.
                        temperature=1.0: unchanged distribution
                        temperature<1.0: more confident/peaked (less random)
                        temperature>1.0: more uniform (more random)
        
        Returns:
            Tensor of shape (batch_size,) or scalar: Sampled token indices
        
        Example:
            >>> logits = model(input_ids)[:, -1, :]  # Get last token logits
            >>> next_token = top_p_sampling(logits, p=0.9, temperature=0.8)
        """
        shape=logits.shape
        if logits.dim()==1:
            logits=logits.unsqueeze(0)
        logits=logits/temperature
        probs=neural_operation.softmax(logits,dim=-1)
        sorted_probs,sorted_indices=torch.sort(probs,descending=True,dim=-1)

        cumulative_probs=torch.cumsum(sorted_probs,dim=-1)
        # filter out all probabilities 
        sorted_indices_remove=cumulative_probs>p
        sorted_indices_remove[::,0]=False
    
        indices_to_remove = sorted_indices_remove.scatter(
            dim=-1, 
            index=sorted_indices, 
            src=sorted_indices_remove
        )
        logits[indices_to_remove] = -math.inf
        
        # Renormalize and sample
        filtered_probs = neural_operation.softmax(logits, dim=-1)
        next_token = torch.multinomial(filtered_probs, num_samples=1)
        
        # Return in original shape
        if len(shape) == 1:
            return next_token.unsqueeze(0)
        return next_token

    @staticmethod
    def top_k_sampling(logits:torch.Tensor,k:int=50,temperature:int=1.0):
        shape=logits.shape
        if logits.dim()==1:
            logits=logits.unsqueeze(0)
        logits=logits/temperature

        top_k_logits,top_k_indices=torch.topk(logits,k=min(k,logits.size(-1)),dim=-1)

        logits_filtered=torch.full_like(logits,math.inf)
        logits_filtered.scatter(dim=-1,index=top_k_indices,src=top_k_logits)

        probs=neural_operation.softmax(logits_filtered,dim=-1)
        next_token=torch.multinomial(probs,num_samples=1)
        if len(shape)==1:
            return next_token.unsqueeze(0)
        return next_token
    




class MultiheadSelfAttention(nn.Module):
    def __init__(
            self, 
            d_model:int,
            num_heads:int,
            rope_theta:float=10000,
            max_seq_len:int=1024,
            layer_idx:int=0,
            device:torch.device=None,
            dtype:torch.dtype=None,
            use_cache:bool=False,
            is_decoder:bool=False
            ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.use_cache=use_cache
        self.layer_idx=layer_idx
        self.W_q=Linear(d_model,d_model,device=device,dtype=dtype)
        self.W_k=Linear(d_model,d_model,device=device,dtype=dtype)
        self.W_v=Linear(d_model,d_model,device=device,dtype=dtype)
        self.W_o=Linear(d_model,d_model,device=device,dtype=dtype)
        self.is_decoder=is_decoder
        self.rope=RoPE(theta=rope_theta,d_k=self.d_head, max_seq_len=max_seq_len,device=device,dtype=dtype)
        
    def forward(
            self,
            x,
            past_kv_values:Optional[DynamicCache]=None,
            cache_position: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor]=None

        ):
        # x=[batch, seq_size, embedding_size]
        proj_q=self.W_q(x)
        proj_k=self.W_k(x)        
        proj_v=self.W_v(x)
        
        proj_Q=rearrange(proj_q,'b s (h d) -> b h s d',h=self.num_heads,d=self.d_head)
        proj_K=rearrange(proj_k,'b s (h d) -> b h s d',h=self.num_heads,d=self.d_head)
        proj_V=rearrange(proj_v,'b s (h d) -> b h s d',h=self.num_heads,d=self.d_head)
        
        # Expand token_positions for all heads if provided
        batch_size, num_heads, seq_len, d_head = proj_Q.shape
        if cache_position is not None:
            # token_positions: [batch, seq] -> [batch*heads, seq]
            token_positions_expanded = position_ids
            token_positions_expanded = token_positions_expanded.expand(
                batch_size, num_heads, seq_len
            ).reshape(batch_size * num_heads, seq_len)
        else:
            token_positions_expanded = None

        proj_Q=self.rope(proj_Q)
        proj_K=self.rope(proj_K)

        if past_kv_values is not None:
            while len(past_kv_values.layers)<=self.layer_idx:
                past_kv_values.layers.append(DynamicLayer())
            
            proj_K,proj_V=past_kv_values.update(
                key_states=proj_K,
                value_states=proj_V,
                layer_idx=self.layer_idx
            )

        L_total = proj_K.size(2)
        L_new = proj_Q.size(2)
        # Finished adding

        causal_mask=None
        if self.is_decoder:
            causal_mask = torch.tril(torch.ones(L_new, L_total, device=x.device, dtype=torch.bool))

            # causal_mask = causal_mask[-seq_len:,:]
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # broadcast to (1,1,seq_len,seq_len)
            
        # tensor_result=neural_operation.scaled_dot_product_attention(proj_Q,proj_K,proj_V,causal_mask)
        tensor_result=neural_operation.scaled_dot_product_attention(proj_Q, proj_K, proj_V, causal_mask)

        output=rearrange(tensor_result, 'batch num_heads seq d_head ->batch seq (num_heads d_head)')
        return self.W_o(output)

class TransformerBlock(nn.Module):
    def __init__(
            self,
            *,
            d_model:int,
            num_heads:int,
            d_ff:int,
            rope_theta:float=1e5,
            max_seq_len:int=1024,
            layer_idx:int=0,
            is_decoder:bool=True,
            device:torch.device=None,
            dtype:torch.dtype=None, 
            ):
        super().__init__()
        self.d_model=d_model # Dimentionality transformer block inputs
        self.num_heads=num_heads
        self.d_ff=d_ff # Dimensionality of the position-wise feed-forward inner layer.
        self.device=device
        self.dtype=dtype
        self.multihead_self_attention=MultiheadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            rope_theta=rope_theta,
            max_seq_len=max_seq_len,
            layer_idx=layer_idx,
            device=device,
            dtype=dtype,
            is_decoder=is_decoder
            )
        self.feed_forward_layer = nn.Sequential(
            Linear(d_model,d_ff),
            swiGLU(d_ff,d_model),
            Linear(d_ff,d_model)
        )
        self.rmsnorm_attn=RMSNorm(d_model=self.d_model,device=device,dtype=dtype)
        self.rmsnorm_ffn=RMSNorm(d_model=self.d_model,device=device,dtype=dtype)
        
    def forward(self, x, past_kv_values,cache_position,position_ids):
        residual=x
        x=self.multihead_self_attention(self.rmsnorm_attn(x),past_kv_values,cache_position,position_ids)
        x+=residual
        residual=x
        x=self.feed_forward_layer(self.rmsnorm_ffn(x))
        x+=residual
        return x
    

# Helper class for VIT
class PatchEmbeddings(nn.Module):
    def __init__(
            self,
            img_size:int=96,
            patch_size:int=16,
            d_model:int=512,
            *,
            device:torch.device,
            dtype:torch.dtype
        ):
        super().__init__()
        self.img_size=img_size
        self.patch_size=patch_size
        self.num_patches=(img_size//patch_size)**2
        self.conv=nn.Conv2d(
            in_channels=3,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size,
            device=device,
            dtype=dtype
        )
    def forward(self, x):
        x=self.conv(x)
        x=x.flatten(2)
        x=x.transpose(1,2)
        return x

# class CLIPVisionTower(nn.Module):
#     def __init__(
#             self, 
#             vision_towe,
#             args,
#         ):
#         super().__init__()
#         self.is_loaded=False
#         self.vision_tower_name=vision_towe
#         self.select_feature=getattr(args, 'mm_vision_select_feature', 'patch')

class VIT(nn.Module):
    def __init__(
            self,
            img_size:int,
            patch_size:int,
            d_model:int,
            num_heads:int,
            rope_theta:int,
            n_blocks: int,
            emb_dropout:float=0.2,
            *,
            device:torch.device,
            dtype:torch.dtype
        ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        self.path_embeding=PatchEmbeddings(
            img_size=img_size,
            patch_size=patch_size,
            d_model=d_model,
            device=device,
            dtype=dtype
        )
        
        w=torch.zeros(1,1,d_model)
        self.cls_token=nn.Parameter(w)

        # Positional embedding
        # n_patches=(img_size//patch_size)**2
        # weights=torch.empty(d_model,n_patches,device=device,dtype=dtype)
        # pos_emb=nn.init.kaiming_uniform_(weights, a=0, mode='fan_in', nonlinearity='linear')
        # self.pos_embedding=nn.Parameter(pos_emb)

        self.dropout=nn.Dropout(emb_dropout)

        self.transformer_blocks=nn.ModuleList()
        for _ in range(n_blocks):
            self.transformer_blocks.append(
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_model,
                    rope_theta=rope_theta,
                    is_decoder=False,
                    device=device,
                    dtype=dtype
                )
            )

        self.layer_norm = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype
        )
    def forward(self, x:torch.Tensor):
        x=self.path_embeding(x)
        cls_token=self.cls_token.expand(x.shape[0],-1,-1)

        x=torch.cat((cls_token,x),dim=1)
        # x+=self.pos_embedding

        # x=self.dropout(x)
        for transformer_block in self.transformer_blocks:
            x=transformer_block(x,past_kv_values=None,cache_position=None,position_ids=None)
        
        x=self.layer_norm(x[:,0])
        return x
    
# class MLPExperts(nn.Module):
#     """
#     SparseMLP MLP perceptron with sparse expert parallel layers
#     """
#     def __init__(
#             self,
#             num_experts:int,
#             d_model:int,
#             intermediate_size:int,
#             expert_parallel: Optional[str]="EP",
#             activation: Optional[Callable]=None,
#             dropout=0.2,
#             gated:Optional[bool]=False,
#             *
#             device:torch.device,
#             dtype:torch.dtype
#         ):
#         super().__init__()
#         self.expert_parallel=expert_parallel
#         self.num_experts=num_experts
#         self.d_model=d_model
#         self.intermediate_size=intermediate_size
#         self.expert_parallel=expert_parallel,
#         self.activation=activation,
#         self.dropout=dropout,
        
#         self.num_local_experts=self.num_experts
#         self.ep_size=1
#         if gated:
#             self.wi_gate=nn.Parameter(
#                 torch.empty(
#                     num_experts,d_model,intermediate_size*2 if activation=='swiglu' else intermediate_size
#                 ),
#             )
#             self.wi_up=nn.Parameter(torch.empty(num_experts,d_model,intermediate_size))
#         self.wo=nn.Parameter(torch.empty(num_experts,d_model,intermediate_size))


#     def forward(self,x):
#         return self.net(x)

class MoEExpert(nn.Module):
    """Single expert - a simple FFN"""
    def __init__(
            self, 
            d_model, 
            h_d=None,
            *,
            device:torch.device,
            dtype:torch.dtype
        ):
        super().__init__()
        h_d = h_d or 4 * d_model
        self.net = nn.Sequential(
            Linear(d_model, h_d,device=device,dtype=dtype),
            nn.ReLU(),
            Linear(h_d, d_model,device=device,dtype=dtype)
        )
    
    def forward(self, x):
        return self.net(x)


class NoisyTopkRouter(nn.Module):
    def __init__(
            self,
            d_model,
            num_experts,
            top_k,
            device:torch.device,
            dtype:torch.dtype
        ):
        super().__init__()
        self.top_k=top_k
        self.topkroute_linear=Linear(d_model,num_experts,dtype=dtype,device=device)
        self.noisy_linear=Linear(d_model,num_experts,dtype=dtype,device=device)

    def forward(self, x):
        b,s,d=x.shape
        logits=self.topkroute_linear(x)

        if self.training:
            noise_logits=self.noisy_linear(x)
            noise=torch.randn_like(logits)*torch.nn.functional.softplus(noise_logits)
            logits=logits+noise

        routing_probs=nn.functional.softmax(logits,dim=-1)
        top_k_probs,top_k_indices = torch.topk(routing_probs,self.top_k,dim=-1)
        top_k_probs=top_k_probs/top_k_probs.sum(dim=-1,keepdim=True)
        return top_k_probs,top_k_indices,routing_probs


class SparseMoE(nn.Module):
    def __init__(
            self,
            d_model,
            num_experts,
            top_k,
            expert_d,
            *,
            device:torch.device,
            dtype:torch.dtype
        ):
        super().__init__()
        self.d_model=d_model
        self.num_experts=num_experts
        self.top_k=top_k
        self.expert_d=expert_d       

        self.router=NoisyTopkRouter(
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k,
            device=device,
            dtype=dtype
        )
        self.experts=nn.ModuleList()
        for i_expert in range(num_experts):
            self.experts.append(
                MoEExpert(
                    d_model=d_model,
                    h_d=expert_d,
                    device=device,
                    dtype=dtype
                )
            )
    def forward(self,x:torch.Tensor):
        b,s,d=x.shape

        top_k_probs,top_k_indices,routing_probs=self.router(x)
    
        output=torch.zeros_like(x)

        flat_x=x.reshape(-1,d)
        flat_probs=top_k_probs.reshape(-1,self.top_k)
        flat_indices = top_k_indices.reshape(-1,self.top_k)
        flat_output=output.reshape(-1,d)

        for expert_idx in range(self.num_experts):
            expert_mask=(flat_indices==expert_idx)

            token_indices=torch.any(expert_mask,dim=-1)
            if not token_indices.any():
                continue

            expert_input=   [token_indices]
            expert_output=self.experts[expert_idx][expert_input]

            expert_proba_mask=expert_mask[token_indices]
            expert_probs=flat_probs[token_indices]

            token_expert_probs=(expert_probs*expert_proba_mask.float()).sum(dim=-1)

            weighted_output=expert_output*token_expert_probs
            flat_output[token_indices]+=weighted_output
        output=flat_output.reshape(b,s,d)
        aux_loss=self.load_balancing_loss(routing_probs)
        return output, aux_loss
    def load_balancing_loss(self, routing_probs):
        """
        Compute load balancing loss to encourage uniform expert utilization
        
        Args:
            routing_probs: (batch, seq, num_experts)
        """
        # Average probability of routing to each expert
        expert_probs = routing_probs.mean(dim=[0, 1])  # (num_experts,)
        
        # Compute coefficient of variation
        # We want all experts to be used equally
        target_prob = 1.0 / self.num_experts
        loss = ((expert_probs - target_prob) ** 2).sum()
        
        return loss
    
class SparseMoEBlock(nn.Module):
    def __init__(
            self,
            d_model,
            num_heads,
            num_experts,
            top_k,
            layer_idx,
            expert_d,
            dropout=0.1,
            is_decoder=False,
            *,
            device:torch.device,
            dtype:torch.dtype
        ):
        super().__init__()
        self.norm1=RMSNorm(d_model=d_model,device=device,dtype=dtype)
        self.attn=MultiheadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            layer_idx=layer_idx,
            device=device,
            dtype=dtype,
            is_decoder=is_decoder
        )
        self.norm2=RMSNorm(d_model=d_model,device=device,dtype=dtype)
        self.sparse_moe=SparseMoE(d_model=d_model,expert_d=expert_d,num_experts=num_experts,top_k=top_k,device=device,dtype=dtype)
    
    def forward(self, x):
        residual=x
        x=self.norm1(x)
        x=self.attn(x)
        x+=residual
        residual=x
        x=self.norm2(x)
        x+=self.sparse_moe(x)
        x+=residual
        return x
    
class AleMoEVLM(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            context_length: int,
            d_model: int,
            top_k: int,
            num_layers: int,
            num_heads: int,
            d_ff: int,
            expert_d:int,
            rope_theta: float,
            image_d_model:int,
            num_experts:int,
            img_size:int=None,
            patch_size:int=None,
            device:torch.device=None,
            dtype:torch.dtype=None,
            use_images:bool=False
        ):
        super().__init__()
        self.vocab_size=vocab_size
        self.context_length=context_length
        self.d_model=d_model
        self.num_layers=num_layers
        self.num_heads=num_heads
        self.d_ff=d_ff
        self.rope_theta=rope_theta
        self.device=device
        self.dtype=dtype
        self.use_images=use_images

        if use_images:
            if not (img_size is not None and patch_size is not None):
                raise ValueError('Missing dimensions for images')
            self.vision_encoder=VIT(
                img_size=img_size,
                patch_size=patch_size,
                d_model=image_d_model,
                num_heads=num_heads,
                rope_theta=rope_theta,
                n_blocks=num_layers,
                device=device,
                dtype=dtype
            )

        self.embeddings=Embedding(num_embeddings=vocab_size,embedding_dim=d_model,device=device,dtype=dtype)
        if use_images:
            self.image_projection=nn.Sequential(
                Linear(image_d_model,d_model,device=device,dtype=dtype)
            )
        
        self.sparse_moe=nn.ModuleList()
        for i_block in range(num_layers):
            self.sparse_moe.append(
                SparseMoEBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    num_experts=num_experts,
                    layer_idx=i_block,
                    expert_d=expert_d,
                    is_decoder=True,
                    top_k=top_k,
                    device=device,
                    dtype=dtype
                )
            )
        
        self.norm1=RMSNorm(d_model=d_model,device=device,dtype=dtype)
        self.lm_head=Linear(d_model,vocab_size,device=device,dtype=dtype)


    def forward(self,x,img_x=None,targets=None):
        x=self.embeddings(x)
        if self.use_images and img_x is not None:
            image_embeds=self.vision_encoder(img_x)
            img_emb=self.image_projection(image_embeds).unsqueeze(1)
            x=torch.cat([x,img_emb],dim=1)
        for i_block in range(self.num_layers):
            x=self.sparse_moe[i_block](x)
        
        x=self.norm1(x)
        logits=self.lm_head(x)
        return logits
    


class CELoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, logits:torch.Tensor, targets: torch.Tensor):
        targets = targets.long()
        shift = torch.max(logits,dim=-1, keepdim=True).values
        logits_shifted= logits-shift
        logsumexp=torch.log(torch.sum(torch.exp(logits_shifted),dim=-1))
        correct=torch.gather(logits_shifted,-1, targets.unsqueeze(-1)).squeeze(-1)
        loss = -(correct - logsumexp)
        return loss.mean()
    