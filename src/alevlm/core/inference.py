from typing import Any, Optional
from abc import ABC, abstractmethod
from collections.abc import Iterable
import torch
class CacheLayer(ABC):
    def __init__(self):
        self.keys:Optional[torch.Tensor]=None
        self.values:Optional[torch.Tensor]=None
        self.is_initialized=False
        self.device='cuda'
    def __repr__(self):
        return f"{self.__class__.__name__}"
    
    @abstractmethod
    def update(
        self,
        key_states:torch.Tensor,
        value_states:torch.Tensor,
        cache_kwargs:Optional[dict[str,Any]]=None
    ):
        ...
    
    def offload(self):
        if self.is_initialized:
            self.keys=self.keys.to('cpu',non_blocking=True)
            self.values=self.values.to('cpu',non_blocking=True)

    def prefetch(self):
        if self.is_initialized and self.keys.device!=self.device:
            self.keys=self.keys.to(self.device,non_blocking=True)
            self.values=self.values.to(self.device,non_blocking=True)

class DynamicLayer(CacheLayer):
    def lazy_initialization(self, key_states:torch.Tensor):
        self.dtype,self.device=key_states.dtype,key_states.device
        self.keys=torch.tensor([],dtype=self.dtype,device=self.device)
        self.values=torch.tensor([],dtype=self.dtype,device=self.device)
        self.is_initialized=True
    def update(self, key_states, value_states, cache_kwargs = None):
        if not self.is_initialized:
            self.lazy_initialization(key_states=key_states)
        self.keys=torch.cat([self.keys,key_states],dim=-2)
        self.values=torch.cat([self.values,value_states],dim=-2)
        return self.keys,self.values
    
    def get_mask_sizes(self,cache_position:torch.Tensor)->tuple[int,int]:
        kv_offset=0
        query_length=cache_position.shape[0]
        kv_length=self.get_seq_length()+query_length
        return kv_length,kv_offset

    def get_seq_length(self)->int:
        """
        batch num_heads seq_lenth head_dim_emb
        """
        if not self.is_initialized or self.keys.numel()==0:
            return 0
        return self.keys.shape[-2]

    def crop(self, max_length:int)-> None:
        if max_length<0:
            max_length=self.get_seq_length()-abs(max_length)
        
        if self.get_seq_length()<=max_length:
            return
        self.keys=self.keys[...,:max_length,:]
        self.values=self.values[...,:max_length,:]

class Cache:
    """
    Supports all the layers' cache
    """
    def __init__(
            self,
            layers:Optional[list[DynamicLayer]]=None,
            layer_class_to_replicate:Optional[type[DynamicLayer]]=None,
            offloading:bool=False,
            offload_only_non_sliding:bool=True,
        ):
        if layers is not None and layer_class_to_replicate is not None:
            raise ValueError('Cant use both layers and layer_class_to_replicate')
        if layers is None and layer_class_to_replicate is None:
            raise ValueError('You should provide exactly one of `layers` or `layer_class_to_replicate` to initialize a Cache.')
        self.layers=layers if layers is not None else []
        self.layer_class_to_replicate=layer_class_to_replicate
        self.offloading=offloading
        if self.offloading:
            self.only_non_sliding=offload_only_non_sliding
            self.prefetch_stream=torch.Stream()

    def prefetch(self, layer_idx:int,only_non_sliding:bool=True):
        with self.prefetch_stream:
            self.layers[layer_idx].prefetch()

    def update(
            self,
            key_states:torch.Tensor,
            value_states:torch.Tensor,
            layer_idx:int,
            cache_kwargs:Optional[dict[str,Any]]=None
        ):
        keys,values=self.layers[layer_idx].update(key_states=key_states,value_states=value_states,cache_kwargs=cache_kwargs)
        if self.offloading:
            self.offloading(layer_idx,self.only_non_sliding)
        return keys,values
    def crop(self,max_lenth:int):
        for layer_idx in range(len(self.layers)):
            self.layers[layer_idx].crop(max_length=max_lenth)
        
    @property
    def cache_len(self) -> int:
        """Return the maximum cache length of the cache"""
        # if len(self.layers)>0:
        values = [layer.get_seq_length() for layer in self.layers]
        return len(values)

    def length_DynamicLayer(self,idx) -> int:
        """Return the maximum cache length of the cache"""
        if len(self.layers)>0:
            dynamiclayer = self.layers[idx]
            return dynamiclayer.get_seq_length()
        else:
            return 0

class DynamicCache(Cache):
    def __init__(
            self,
            ddp_cache_data:Optional[Iterable[tuple[Optional[torch.Tensor],...]]]=None,
            config:Optional[dict]=False,
            offloading:bool=False,
            offload_only_non_sliding:bool=False
        ):
        layers=[]
        if ddp_cache_data is not None:
            for layer_idx,kv_and_optional_sliding in enumerate(ddp_cache_data):
                if config is None:
                    sliding_window_tensor=kv_and_optional_sliding[2] if len(kv_and_optional_sliding)==3 else None
                    if sliding_window_tensor is not None:
                        sliding_window=sliding_window_tensor[0].item()
                        # TODO
                        layers.append(DynamicLayer())
                        raise NotImplementedError('My bad dog')
                    else:
                        layers.append(DynamicLayer())
                _,_=layers[layer_idx].update(kv_and_optional_sliding[0],kv_and_optional_sliding[1])
        
        if len(layers)==0:
            super().__init__(
                layer_class_to_replicate=DynamicLayer,
                offloading=offloading,
                offload_only_non_sliding=offload_only_non_sliding
            )
        else:
            super().__init__(layers=layers,offloading=offloading,offload_only_non_sliding=offload_only_non_sliding)
    
    def __iter__(self):
        for layer in self.layers:
            yield layer.keys, layer.values,getattr(layer,"_sliding_window_tensor",None)