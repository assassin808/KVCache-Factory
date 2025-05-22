import copy
import importlib.metadata
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from packaging import version

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import is_hqq_available, is_quanto_available, is_torchdynamo_compiling, logging


import math
import torch.nn.functional as F
import torch.nn as nn


if is_quanto_available():
    quanto_version = version.parse(importlib.metadata.version("quanto"))
    if quanto_version >= version.parse("0.2.0"):
        from quanto import AffineQuantizer, MaxOptimizer, qint2, qint4

if is_hqq_available():
    from hqq.core.quantize import Quantizer as HQQQuantizer

logger = logging.get_logger(__name__)


class Cache(torch.nn.Module):
    """
    Base, abstract class for all caches. The actual data structure is specific to each subclass.
    """

    def __init__(self):
        super().__init__()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. These are specific to each subclass and allow new types of
                cache to be created.

        Return:
            A tuple containing the updated key and value states.
        """
        raise NotImplementedError("Make sure to implement `update` in a subclass.")

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        raise NotImplementedError("Make sure to implement `get_seq_length` in a subclass.")

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states, if there is any."""
        raise NotImplementedError("Make sure to implement `get_max_length` in a subclass.")

    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = 0) -> int:
        """Given the sequence length of the new inputs, returns the usable length of the cache."""
        # Cache without size limit -> all cache is usable
        # Cache with size limit -> if the length cache plus the length of the new inputs is larger the maximum cache
        #   length, we will need to evict part of the cache (and thus not all cache is usable)
        max_length = self.get_max_length()
        previous_seq_length = self.get_seq_length(layer_idx)
        if max_length is not None and previous_seq_length + new_seq_length > max_length:
            return max_length - new_seq_length
        return previous_seq_length

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.key_cache)):
            device = self.key_cache[layer_idx].device
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
            device = self.value_cache[layer_idx].device
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))

    @property
    def seen_tokens(self):
        logger.warning_once(
            "The `seen_tokens` attribute is deprecated and will be removed in v4.41. Use the `cache_position` "
            "model input instead."
        )
        if hasattr(self, "_seen_tokens"):
            return self._seen_tokens
        else:
            return None


@dataclass
class CacheConfig:
    """
    Base class for cache configs
    """

    cache_implementation: None

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        """
        Constructs a CacheConfig instance from a dictionary of parameters.
        Args:
            config_dict (Dict[str, Any]): Dictionary containing configuration parameters.
            **kwargs: Additional keyword arguments to override dictionary values.

        Returns:
            CacheConfig: Instance of CacheConfig constructed from the dictionary.
        """
        config = cls(**config_dict)
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)
        return config

    # Copied from transformers.utils.quantization_config.QuantizationConfigMixin.to_json_file
    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default
                `QuantizationConfig()` is serialized to JSON file.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            config_dict = self.to_dict()
            json_string = json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

            writer.write(json_string)

    # Copied from transformers.utils.quantization_config.QuantizationConfigMixin.to_dict
    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary. Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        return copy.deepcopy(self.__dict__)

    # Copied from transformers.utils.quantization_config.QuantizationConfigMixin.__iter__
    def __iter__(self):
        """allows `dict(obj)` for situations where obj may be a dict or QuantizationConfigMixin"""
        for attr, value in copy.deepcopy(self.__dict__).items():
            yield attr, value

    # Copied from transformers.utils.quantization_config.QuantizationConfigMixin.__repr__
    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"

    def to_json_string(self):
        """
        Serializes this instance to a JSON formatted string.
        Returns:
            str: JSON formatted string representing the configuration instance.
        """
        return json.dumps(self.__dict__, indent=2) + "\n"

    # Copied from transformers.utils.quantization_config.QuantizationConfigMixin.update
    def update(self, **kwargs):
        """
        Updates attributes of this class instance with attributes from `kwargs` if they match existing attributes,
        returning all the unused kwargs.

        Args:
            kwargs (`Dict[str, Any]`):
                Dictionary of attributes to tentatively update this class.

        Returns:
            `Dict[str, Any]`: Dictionary containing all the key-value pairs that were not used to update the instance.
        """
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                to_remove.append(key)

        # Remove all the attributes that were updated, without modifying the input dict
        unused_kwargs = {key: value for key, value in kwargs.items() if key not in to_remove}
        return unused_kwargs

class PretrainedConfig: # Dummy
    def __init__(self, num_hidden_layers=32, num_attention_heads=32, hidden_size=4096):
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        # Add other attributes like kv_cluster if used by your init_snapkv
        # self.kv_cluster = type('KVCluster', (), {'max_capacity_prompt': 256})()


class DynamicCache(Cache):
    def __init__(self, config: PretrainedConfig = None, sink_size: int = 8, window_size: int = 8) -> None:
        super().__init__()
        self.config = config if config is not None else PretrainedConfig()
        self.num_hidden_layers = self.config.num_hidden_layers
        self.num_heads = self.config.num_attention_heads
        self.head_dim = self.config.hidden_size // self.config.num_attention_heads
        self._seen_tokens = 0
        
        self.sink_size = sink_size
        self.window_size = window_size

        self.prefill_len = 0 # Not directly used in the logic below but present in your class

        # These will store the final structured K/V caches [B, H, S_layer, D]
        self.retained_key_cache: List[Optional[torch.Tensor]] = [None] * self.num_hidden_layers
        self.retained_value_cache: List[Optional[torch.Tensor]] = [None] * self.num_hidden_layers
        # key_unit_cache can store Tensors or Reuse Dictionaries
        self.key_unit_cache: List[Optional[Union[torch.Tensor, Dict]]] = [None] * self.num_hidden_layers
        self.value_unit_cache: List[Optional[torch.Tensor]] = [None] * self.num_hidden_layers
        
        # Temporary caches during prefill, before restructuring
        self._original_key_cache: List[torch.Tensor] = []
        self._original_value_cache: List[torch.Tensor] = []

        # For importance scoring and reuse logic
        self.query_cache: List[Optional[torch.Tensor]] = [None] * self.num_hidden_layers
        self.decode_q: List[Optional[torch.Tensor]] = [None] * self.num_hidden_layers
        self.layer_map: Dict[int, Dict[int, Tuple[int, int, float]]] = {} # target_l -> {target_h -> (anchor_l, anchor_h, scale)}
        
        # Temporarily stores (unit_indices_abs, retained_core_indices_abs) per layer during restructuring
        # Each element: Tuple[Tensor[B,H,N_unit], Tensor[B,H,N_core]]
        self.indices_analysis_results: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * self.num_hidden_layers


    def _prepare_device_dtype_from_states(self, key_states):
        return key_states.device, key_states.dtype

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
        hidden_states: Optional[torch.Tensor] = None, # Passed as hidden_states_input in my prev example
        query_states: Optional[torch.Tensor] = None, # Passed as query_states_input
        attention_mask = None, # Passed as attention_mask_input
        max_len: int = 256, # Max sequence length for f() scaling function
        ratio: float = 0.5, # Ratio for important tokens
        is_profiling: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, None]:

        device, dtype = self._prepare_device_dtype_from_states(key_states)

        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Accumulate into temporary original full caches during prefill
        if len(self._original_key_cache) <= layer_idx:
            self._original_key_cache.append(key_states)
            self._original_value_cache.append(value_states)
        else:
            self._original_key_cache[layer_idx] = torch.cat([self._original_key_cache[layer_idx], key_states], dim=-2)
            self._original_value_cache[layer_idx] = torch.cat([self._original_value_cache[layer_idx], value_states], dim=-2)
        
        # Calculate and store query_cache for importance scoring
        current_layer_accumulated_keys = self._original_key_cache[layer_idx]
        query_states = query_states.transpose(1, 2)
        if query_states is not None and query_states.shape[-2] >= self.window_size:
            q_for_importance = query_states[:, :, -self.window_size:, :]
            num_accumulated_keys = current_layer_accumulated_keys.shape[-2]
            
            if num_accumulated_keys > self.sink_size + self.window_size:
                k_target_for_importance = current_layer_accumulated_keys[:, :, self.sink_size : num_accumulated_keys - self.window_size, :]
                if k_target_for_importance.shape[-2] > 0: # Ensure middle section is not empty
                    # print(q_for_importance.shape,k_target_for_importance.transpose(2, 3).shape)
                    attn_logits = torch.matmul(q_for_importance, k_target_for_importance.transpose(2, 3)) / math.sqrt(q_for_importance.shape[-1])
                    attn_weights = nn.functional.softmax(attn_logits, dim=-1, dtype=torch.float32).to(dtype)
                    self.query_cache[layer_idx] = attn_weights.sum(dim=-2) # Sum over query head dimension: [B, H, S_middle]
                elif self.query_cache[layer_idx] is None:
                     self.query_cache[layer_idx] = torch.empty((key_states.shape[0], self.num_heads, 0), device=device, dtype=dtype)
            elif self.query_cache[layer_idx] is None :
                 self.query_cache[layer_idx] = torch.empty((key_states.shape[0], self.num_heads, 0), device=device, dtype=dtype)

        # --- Profiling Mode (Generates layer_map_final.csv) ---
        if is_profiling:
            if layer_idx == self.num_hidden_layers - 1:
                print("Starting profiling to generate layer_map_final.csv...")
                # Placeholder for your detailed profiling logic.
                # This logic calculates similarities between (Q_i, K_i) and (Q_j, K_j) patterns.
                # It needs careful handling of which Q and K are used (raw queries vs. summed weights from query_cache).
                # The output should be `profiling_results_map` list of tuples.
                profiling_results_map = [] 
                # Example dummy entry, replace with actual calculation
                # if self.num_hidden_layers > 1 and self.num_heads > 1:
                #    profiling_results_map.append((0,1,0,0,0,0.95, 0.8)) # (i,j,seg,hi,hj,sim,scale)

                # --- Begin Your Profiling Logic Adapted ---
                # This assumes self.query_cache (summed weights) and self._original_key_cache (full keys)
                # are suitable for the similarity calculation defined in your original snippet.
                # The original snippet used QK products. If self.query_cache holds Q_states for profiling:
                # for i in range(self.num_hidden_layers):
                #     # prev_segment = torch.matmul(Q_FOR_PROFILING[i], self._original_key_cache[i].transpose(2,3)) ...
                # This part is highly specific to how profiling Qs are obtained.
                # --- End Your Profiling Logic Adapted ---
                
                print(f"Generated {len(profiling_results_map)} potential reuse pairs before filtering.")
                # Filtering logic (used_segment, replaced_segment)
                final_map_for_csv = []
                # ... (your filtering logic here based on used_segment, replaced_segment) ...
                # For now, let's assume final_map_for_csv is populated by your logic.
                final_map_for_csv = profiling_results_map # Without filtering for this placeholder

                with open('layer_map_final.csv', 'w') as f:
                    for item_map in final_map_for_csv:
                        f.write(','.join([str(x) for x in item_map]) + '\n')
                print("Profiling finished, layer_map_final.csv written. Exiting.")
                exit(0)
            return key_states, value_states, None # Standard return during profiling

        # --- Cache Restructuring (End of Prefill, layer_idx == N-1, not profiling) ---
        if layer_idx == self.num_hidden_layers - 1:
            print("Restructuring KV cache for true reuse...")
            # 1. Read layer_map_final.csv into self.layer_map
            self.layer_map = {}
            try:
                with open('layer_map_final.csv', 'r') as f:
                    for line in f:
                        parts = line.strip().split(',')
                        source_l, target_l, _, source_h, target_h, _, scale_val = (
                            int(parts[0]), int(parts[1]), int(parts[2]),
                            int(parts[3]), int(parts[4]),
                            float(parts[5]), float(parts[6])
                        )
                        if target_l not in self.layer_map: self.layer_map[target_l] = {}
                        self.layer_map[target_l][target_h] = (source_l, source_h, scale_val)
            except FileNotFoundError:
                print("Warning: layer_map_final.csv not found. No KV reuse will be configured.")

            # 2. Attention-based Index Selection for ALL layers
            f_scaled_size = lambda r, ml: int(ml / (10/32 + 22/32 * (r + 1) / 2)) # Your f()
            
            max_retained_core_tokens_layer = [0] * self.num_hidden_layers
            max_unit_tokens_layer = [0] * self.num_hidden_layers

            for i_layer in range(self.num_hidden_layers):
                if self.query_cache[i_layer] is None or self._original_key_cache[i_layer] is None:
                    self.indices_analysis_results[i_layer] = (
                        torch.empty((key_states.shape[0], self.num_heads, 0), dtype=torch.long, device=device),
                        torch.empty((key_states.shape[0], self.num_heads, 0), dtype=torch.long, device=device)
                    )
                    continue

                attn_weights_sum = self.query_cache[i_layer] # [B, H, S_middle]
                s_middle = attn_weights_sum.shape[-1]
                
                current_total_seq_len = self._original_key_cache[i_layer].shape[-2]
                target_total_compressed_len = min(f_scaled_size(ratio, max_len), current_total_seq_len)
                
                num_middle_tokens_to_keep_overall = max(0, target_total_compressed_len - self.window_size - self.sink_size)
                k_all = min(num_middle_tokens_to_keep_overall, s_middle)

                if k_all <= 0:
                    all_indices_rel = torch.empty((key_states.shape[0], self.num_heads, 0), dtype=torch.long, device=device)
                else:
                    all_indices_rel = attn_weights_sum.topk(k_all, dim=-1).indices # Relative to middle section

                num_retained_core_total = int(target_total_compressed_len * ratio)
                num_retained_core_middle = max(0, num_retained_core_total - self.window_size - self.sink_size)
                k_most_important = min(num_retained_core_middle, k_all)

                if k_most_important > 0 and k_all > 0:
                    gathered_att_values = torch.gather(attn_weights_sum, -1, all_indices_rel)
                    indices_within_all = gathered_att_values.topk(k_most_important, dim=-1).indices
                    retained_core_indices_rel = torch.gather(all_indices_rel, -1, indices_within_all)
                else:
                    retained_core_indices_rel = torch.empty((key_states.shape[0], self.num_heads, 0), dtype=torch.long, device=device)
                
                # Unit indices (those in all_indices_rel but not retained_core_indices_rel)
                # Using your original method for finding unit indices more directly:
                if k_all > 0 :
                    final_exp = retained_core_indices_rel.unsqueeze(-1) # B,H,K_core,1
                    all_exp_u = all_indices_rel.unsqueeze(-2)      # B,H,1,K_all
                    mask_u = (all_exp_u == final_exp).any(dim=-2)  # B,H,K_all (True if in core)
                    
                    # unit_indices_rel = all_indices_rel[~mask_u].reshape(key_states.shape[0], self.num_heads, -1) # This reshape is tricky
                    # A robust way to implement the gather for variable numbers:
                    bs_tmp, h_tmp, _ = all_indices_rel.shape
                    unit_indices_list = []
                    current_max_k_unit = 0
                    for b_ in range(bs_tmp):
                        for h_ in range(h_tmp):
                            selected = all_indices_rel[b_,h_][~mask_u[b_,h_]]
                            unit_indices_list.append(selected)
                            current_max_k_unit = max(current_max_k_unit, selected.shape[0])
                    
                    padded_unit_indices = torch.zeros((bs_tmp, h_tmp, current_max_k_unit), dtype=torch.long, device=device)
                    list_idx = 0
                    for b_ in range(bs_tmp):
                        for h_ in range(h_tmp):
                            s = unit_indices_list[list_idx].shape[0]
                            padded_unit_indices[b_,h_,:s] = unit_indices_list[list_idx]
                            list_idx +=1
                    unit_indices_rel = padded_unit_indices
                else:
                    unit_indices_rel = torch.empty((key_states.shape[0], self.num_heads, 0), dtype=torch.long, device=device)


                self.indices_analysis_results[i_layer] = (unit_indices_rel + self.sink_size, retained_core_indices_rel + self.sink_size)
                max_retained_core_tokens_layer[i_layer] = retained_core_indices_rel.shape[-1]
                max_unit_tokens_layer[i_layer] = unit_indices_rel.shape[-1]

            # 3. Populate Structured Caches (now producing [B,H,S_layer,D] tensors by padding)
            for l_idx in range(self.num_hidden_layers):
                if self._original_key_cache[l_idx] is None: continue

                orig_k_layer = self._original_key_cache[l_idx] # B,H,S_orig,D
                orig_v_layer = self._original_value_cache[l_idx] # B,H,S_orig,D
                s_orig = orig_k_layer.shape[2]
                bs, _, _, d_head = orig_k_layer.shape

                unit_indices_abs, retained_core_indices_abs = self.indices_analysis_results[l_idx]
                
                # Determine padding lengths for this layer
                num_sink = min(self.sink_size, s_orig)
                num_window = min(self.window_size, s_orig - num_sink) # Window comes after sink
                
                # Max number of tokens selected by attention analysis for this layer
                # These are already calculated: max_retained_core_tokens_layer[l_idx], max_unit_tokens_layer[l_idx]
                # Total retained length for this layer: sink + window + padded_retained_core
                s_retained_final_layer = num_sink + num_window + max_retained_core_tokens_layer[l_idx]
                s_unit_final_layer = max_unit_tokens_layer[l_idx]

                # Initialize final cache tensors for this layer
                final_retained_k = torch.zeros((bs, self.num_heads, s_retained_final_layer, d_head), device=device, dtype=dtype)
                final_retained_v = torch.zeros((bs, self.num_heads, s_retained_final_layer, d_head), device=device, dtype=dtype)
                final_unit_k_tensor = torch.zeros((bs, self.num_heads, s_unit_final_layer, d_head), device=device, dtype=dtype)
                final_unit_v_tensor = torch.zeros((bs, self.num_heads, s_unit_final_layer, d_head), device=device, dtype=dtype)
                
                is_layer_reused_unit_k = False # Flag if any head in this layer reuses unit_k

                for h_idx in range(self.num_heads):
                    # Absolute indices for this head
                    h_unit_abs = unit_indices_abs[:, h_idx, :].clamp(0, s_orig - 1) # B, N_unit_head
                    h_retained_core_abs = retained_core_indices_abs[:, h_idx, :].clamp(0, s_orig - 1) # B, N_core_head

                    # Create combined retained indices for this head (sink + window + core)
                    sink_idx_h = torch.arange(0, num_sink, device=device).unsqueeze(0).expand(bs, -1)
                    win_idx_h = torch.arange(s_orig - num_window, s_orig, device=device).unsqueeze(0).expand(bs, -1)
                    
                    combined_retained_indices_h = torch.cat([sink_idx_h, win_idx_h, h_retained_core_abs], dim=-1).unique(dim=-1) # B, N_ret_combined_head
                    
                    # Pad combined_retained_indices_h to s_retained_final_layer before gather (if needed, but gather handles variable indices)
                    # Pad the gathered K/V data before putting into final_retained_k/v

                    # Gather for retained
                    if combined_retained_indices_h.numel() > 0:
                        gathered_k_ret = torch.gather(orig_k_layer[:,h_idx], 1, combined_retained_indices_h.unsqueeze(-1).expand(-1,-1,d_head))
                        gathered_v_ret = torch.gather(orig_v_layer[:,h_idx], 1, combined_retained_indices_h.unsqueeze(-1).expand(-1,-1,d_head))
                        final_retained_k[:, h_idx, :gathered_k_ret.shape[1], :] = gathered_k_ret
                        final_retained_v[:, h_idx, :gathered_v_ret.shape[1], :] = gathered_v_ret
                    
                    # Handle unit_k: direct gather or mark for reuse
                    if l_idx in self.layer_map and h_idx in self.layer_map[l_idx]:
                        # This head reuses its unit_k. The layer's key_unit_cache will store a dict.
                        is_layer_reused_unit_k = True 
                        # No need to populate final_unit_k_tensor for this head if entire layer points to dict
                    elif h_unit_abs.numel() > 0:
                        gathered_k_unit = torch.gather(orig_k_layer[:,h_idx], 1, h_unit_abs.unsqueeze(-1).expand(-1,-1,d_head))
                        final_unit_k_tensor[:, h_idx, :gathered_k_unit.shape[1], :] = gathered_k_unit
                    
                    # Unit_v always local
                    if h_unit_abs.numel() > 0:
                        gathered_v_unit = torch.gather(orig_v_layer[:,h_idx], 1, h_unit_abs.unsqueeze(-1).expand(-1,-1,d_head))
                        final_unit_v_tensor[:, h_idx, :gathered_v_unit.shape[1], :] = gathered_v_unit

                self.retained_key_cache[l_idx] = final_retained_k
                self.retained_value_cache[l_idx] = final_retained_v
                self.value_unit_cache[l_idx] = final_unit_v_tensor
                
                if is_layer_reused_unit_k:
                    # If any head reuses, the layer's entry is a reference map for unit_k.
                    # This implies that if one head reuses, all heads' unit_k in that layer are determined by reuse_map.
                    # This is a simplification. A more granular approach would be per-head dict/tensor.
                    # For now, if layer_map[l_idx] exists, self.key_unit_cache[l_idx] becomes a dict of that.
                    # This means ALL heads in l_idx must follow reuse if any are specified.
                    # A better structure: self.key_unit_cache[l_idx][h_idx] can be tensor or dict.
                    # The current self.key_unit_cache is List[Tensor/Dict]. If it's Dict, it's for the whole layer.
                    if l_idx in self.layer_map:
                         self.key_unit_cache[l_idx] = self.layer_map[l_idx] # Store reuse info for WHOLE layer
                    else: # Should not happen if is_layer_reused_unit_k is True
                         self.key_unit_cache[l_idx] = final_unit_k_tensor
                else:
                    self.key_unit_cache[l_idx] = final_unit_k_tensor
            
            del self._original_key_cache
            del self._original_value_cache
            self.indices_analysis_results = [None] * self.num_hidden_layers # Clear temp list
            torch.cuda.empty_cache()
            print("Cache restructuring complete.")

        return key_states, value_states, None # Return current K/V, not historical

    def update_miniCache_decode(self,
        key_states: torch.Tensor, # New K for current layer, current token: [B, H, 1, D]
        value_states: torch.Tensor,
        layer_idx: int,
        num_layers: int, # Unused, use self.num_hidden_layers
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2] # Should be 1

        # Append new K/V to the end of the existing retained cache for this layer
        if self.retained_key_cache[layer_idx] is not None:
            self.retained_key_cache[layer_idx] = torch.cat([self.retained_key_cache[layer_idx], key_states], dim=-2)
            self.retained_value_cache[layer_idx] = torch.cat([self.retained_value_cache[layer_idx], value_states], dim=-2)
        else:
            # Should not happen if prefill occurred. Initialize if it's the first token ever.
            self.retained_key_cache[layer_idx] = key_states
            self.retained_value_cache[layer_idx] = value_states
        
        # The unit caches are static during decode after prefill restructuring.
        return key_states, value_states, # Return current K/V

    def get_retained_kv(self, layer_idx: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self.retained_key_cache[layer_idx], self.retained_value_cache[layer_idx]

    def get_unit_kv(self, layer_idx: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        unit_k_entry = self.key_unit_cache[layer_idx]
        unit_v_tensor = self.value_unit_cache[layer_idx]
        
        if isinstance(unit_k_entry, dict): # Reuse map for the whole layer
            # All heads in this layer reuse their unit_k based on map
            # Need to construct the full [B,H,S_unit,D] tensor
            # Assume S_unit is consistent for all anchor sources for this layer's unit parts
            # This part is complex if S_unit varies across anchors.
            # For simplicity, assume the reuse dict applies uniformly or we fetch per head from anchor's unit tensor.
            
            # Let's assume anchors (source_l) have their unit_k as Tensors [B,H,S_anchor_unit,D]
            # And the reuse map tells us for each target_h which (source_l, source_h, scale) to use.
            
            # Determine batch_size, num_heads, S_unit_target, head_dim
            # S_unit_target should be max_unit_tokens_layer[layer_idx] from restructuring
            # This info isn't directly stored. Need a way to know target S_unit.
            # For now, let's assume the anchor's S_unit is what we use.
            # This means the S_unit for the target layer might be variable if anchors have different S_units.
            
            # This requires a per-head construction for the unit_k if sources vary.
            # A simpler model: if key_unit_cache[layer_idx] is a dict, it means ALL heads of this layer
            # reuse from a single, common anchor layer (not per-head anchor). This is too simple.
            
            # Corrected logic: iterate target heads, get their anchor, fetch anchor's unit_k for that head, scale.
            # Then stack these to form the final unit_k tensor for the target layer.
            
            bs, _, _, d_head = self.retained_key_cache[layer_idx].shape # Get dims
            # What is S_unit for this layer? It was max_unit_tokens_layer[l_idx] during padding.
            # This length info needs to be accessible, or inferred from anchors.
            # Let's assume the length of the key_unit_cache of the *first* anchor head dictates S_unit for all.
            
            first_target_h = list(unit_k_entry.keys())[0]
            anchor_l, anchor_h_first_anchor, scale_first = unit_k_entry[first_target_h]
            
            anchor_unit_k_tensor_for_first = self.key_unit_cache[anchor_l] # This MUST be a tensor [B,H,S_anchor,D]
            if not isinstance(anchor_unit_k_tensor_for_first, torch.Tensor):
                print(f"ERROR: Anchor layer {anchor_l} for unit_k reuse does not have a tensor unit_k.")
                return None, unit_v_tensor # Or empty tensor

            s_unit_target = anchor_unit_k_tensor_for_first.shape[2]
            final_unit_k = torch.zeros((bs, self.num_heads, s_unit_target, d_head),
                                       device=anchor_unit_k_tensor_for_first.device,
                                       dtype=anchor_unit_k_tensor_for_first.dtype)

            for target_h_idx, (anc_l, anc_h_idx, scale_val) in unit_k_entry.items():
                anchor_layer_unit_k_tensor = self.key_unit_cache[anc_l] # [B,H,S_anchor_unit,D]
                if isinstance(anchor_layer_unit_k_tensor, torch.Tensor) and \
                   anchor_layer_unit_k_tensor.shape[2] == s_unit_target : # Ensure consistent S_unit from anchors
                    final_unit_k[:, target_h_idx, :, :] = anchor_layer_unit_k_tensor[:, anc_h_idx, :, :] * scale_val
                else:
                    print(f"Warning: Mismatch or non-tensor anchor for unit_k head {target_h_idx} in layer {layer_idx}")
            
            return final_unit_k, unit_v_tensor
            
        else: # It's a tensor or None
            return unit_k_entry, unit_v_tensor

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.retained_key_cache)

     
    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "MiniCache":
        """Converts a cache in the legacy cache format into an equivalent `MiniCache`. Used for
        backward compatibility."""
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                cache.retained_key_cache.append(past_key_values[layer_idx][0])
                cache.retained_value_cache.append(past_key_values[layer_idx][1])
                cache.key_unit_cache.append(past_key_values[layer_idx][2])
                cache.value_unit_cache.append(past_key_values[layer_idx][3])
                cache.layer_map = past_key_values[layer_idx][4]
                cache.indices_analysis_results.append(past_key_values[layer_idx][5])


        return cache
    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if len(self.retained_key_cache) <= layer_idx:
            return 0
        return self.retained_key_cache[layer_idx].shape[-2] + self.retained_value_cache[layer_idx].shape[-2]
    
    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length."""
        return None

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format. Used for
        backward compatibility."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            # print(len(self.retained_key_cache),len(self.value_unit_cache),)
            legacy_cache += ((self.retained_key_cache[layer_idx],  self.retained_value_cache[layer_idx], self.key_unit_cache[layer_idx], self.value_unit_cache[layer_idx], self.layer_map, self.indices_analysis_results[layer_idx],),)
        return legacy_cache

    def crop(self, max_length: int):
        """Crop the past key values up to a new `max_length` in terms of tokens. `max_length` can also be
        negative to remove `max_length` tokens. This is used in assisted decoding and contrastive search."""
        # In case it is negative
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)

        if self.get_seq_length() <= max_length:
            return

        self._seen_tokens = max_length
        for idx in range(len(self.key_cache)):
            self.key_cache[idx] = self.key_cache[idx][..., :max_length, :]
            self.value_cache[idx] = self.value_cache[idx][..., :max_length, :]

    def batch_split(self, full_batch_size: int, split_size: int) -> List["DynamicCache"]:
        """Split the current instance into a list of `DynamicCache` by the batch size. This will be used by
        `_split_model_inputs()` in `generation.utils`"""
        out = []
        for i in range(0, full_batch_size, split_size):
            current_split = DynamicCache()
            current_split._seen_tokens = self._seen_tokens
            current_split.key_cache = [tensor[i : i + split_size] for tensor in self.key_cache]
            current_split.value_cache = [tensor[i : i + split_size] for tensor in self.value_cache]
            out.append(current_split)
        return out

    @classmethod
    def from_batch_splits(cls, splits: List["DynamicCache"]) -> "DynamicCache":
        """This is the opposite of the above `batch_split()` method. This will be used by `stack_model_outputs` in
        `generation.utils`"""
        cache = cls()
        for idx in range(len(splits[0])):
            layer_keys = torch.cat([current.key_cache[idx] for current in splits], dim=0)
            layer_values = torch.cat([current.value_cache[idx] for current in splits], dim=0)
            cache.update(layer_keys, layer_values, idx)
        return cache

    def batch_repeat_interleave(self, repeats: int):
        """Repeat the cache `repeats` times in the batch dimension. Used in contrastive search."""
        for layer_idx in range(len(self)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx].repeat_interleave(repeats, dim=0)
            self.value_cache[layer_idx] = self.value_cache[layer_idx].repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor):
        """Only keep the `indices` in the batch dimension of the cache. Used in contrastive search."""
        for layer_idx in range(len(self)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx][indices, ...]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][indices, ...]

class StaticCache(Cache):
    """
    Static Cache class to be used with `torch.compile(model)` and `torch.export()`.

    Parameters:
        config (`PretrainedConfig`):
            The configuration file defining the shape-related attributes required to initialize the static cache.
        max_batch_size (`int`):
            The maximum batch size with which the model will be used.
        max_cache_len (`int`):
            The maximum sequence length with which the model will be used.
        device (`torch.device`):
            The device on which the cache should be initialized. Should be the same as the layer.
        dtype (*optional*, defaults to `torch.float32`):
            The default `dtype` to use when initializing the layer.

    Example:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM, StaticCache

        >>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
        >>> tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

        >>> inputs = tokenizer(text="My name is GPT2", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> # Leave empty space for 10 new tokens, which can be used when calling forward iteratively 10 times to generate
        >>> max_generated_length = inputs.input_ids.shape[1] + 10
        >>> past_key_values = StaticCache(config=model.config, max_batch_size=1, max_cache_len=max_generated_length, device=model.device, dtype=model.dtype)
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> past_kv_length = outputs.past_key_values # access cache filled with key/values from generation
        ```
    """

    def __init__(self, config: PretrainedConfig, max_batch_size: int, max_cache_len: int, device, dtype=None) -> None:
        super().__init__()
        self.max_batch_size = max_batch_size
        self.max_cache_len = config.max_position_embeddings if max_cache_len is None else max_cache_len
        # Some model define a custom `head_dim` != config.hidden_size // config.num_attention_heads
        self.head_dim = (
            config.head_dim if hasattr(config, "head_dim") else config.hidden_size // config.num_attention_heads
        )

        self.dtype = dtype if dtype is not None else torch.float32
        self.num_key_value_heads = (
            config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        )

        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        # Note: There will be significant perf decrease if switching to use 5D tensors instead.
        cache_shape = (max_batch_size, self.num_key_value_heads, self.max_cache_len, self.head_dim)
        for idx in range(config.num_hidden_layers):
            new_layer_key_cache = torch.zeros(cache_shape, dtype=self.dtype, device=device)
            new_layer_value_cache = torch.zeros(cache_shape, dtype=self.dtype, device=device)
            # Notes:
            # 1. `mark_static_address` is used to tag the cache as an fixed data pointer, preventing cuda graph
            #     breaks when updating the cache. It can't be used if the cache code is being compiled (but in that case
            #     it is not needed anyway)
            # 2. `torch.export()` requires mutations to be registered as buffers.
            if not is_torchdynamo_compiling():
                self.register_buffer(f"key_cache_{idx}", torch.zeros(cache_shape, dtype=dtype, device=device))
                self.register_buffer(f"value_cache_{idx}", torch.zeros(cache_shape, dtype=dtype, device=device))
                new_layer_key_cache = getattr(self, f"key_cache_{idx}")
                new_layer_value_cache = getattr(self, f"value_cache_{idx}")
                torch._dynamo.mark_static_address(new_layer_key_cache)
                torch._dynamo.mark_static_address(new_layer_value_cache)
            self.key_cache.append(new_layer_key_cache)
            self.value_cache.append(new_layer_value_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.
        It is VERY important to index using a tensor, otherwise you introduce a copy to the device.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. The `StaticCache` needs the `cache_position` input
                to know how where to write in the cache.

        Return:
            A tuple containing the updated key and value states.
        """
        cache_position = cache_kwargs.get("cache_position")
        self.key_cache[layer_idx] = self.key_cache[layer_idx].to(device=key_states.device)
        self.value_cache[layer_idx] = self.value_cache[layer_idx].to(device=value_states.device)
        k_out = self.key_cache[layer_idx]
        v_out = self.value_cache[layer_idx]

        if cache_position is None:
            k_out.copy_(key_states)
            v_out.copy_(value_states)
        else:
            # Note: here we use `tensor.index_copy_(dim, index, tensor)` that is equivalent to
            # `tensor[:, :, index] = tensor`, but the first one is compile-friendly and it does explicitly an in-place
            # operation, that avoids copies and uses less memory.
            try:
                k_out.index_copy_(2, cache_position, key_states)
                v_out.index_copy_(2, cache_position, value_states)
            except NotImplementedError:
                # The operator 'aten::index_copy.out' is not currently implemented for the MPS device.
                k_out[:, :, cache_position] = key_states
                v_out[:, :, cache_position] = value_states

        return k_out, v_out

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states that were seen by the model."""
        # Occupied cache == any slot in the 3rd dim (sequence length) holds a non-zero value. To save on compute, let's
        # limit the check to the first batch member and head dimension.
        # TODO: deprecate this function in favor of `cache_position`
        return (self.key_cache[layer_idx][0, 0].any(dim=-1)).sum()

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states."""
        return self.max_cache_len

    def reset(self):
        """Resets the cache values while preserving the objects"""
        for layer_idx in range(len(self.key_cache)):
            # In-place ops prevent breaking the static address
            self.key_cache[layer_idx].zero_()
            self.value_cache[layer_idx].zero_()
