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

class DynamicCache(Cache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.

    Example:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

        >>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
        >>> tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

        >>> inputs = tokenizer(text="My name is GPT2", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> past_key_values = DynamicCache()
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> past_kv_length = outputs.past_key_values # access cache filled with key/values from generation
        ```
    """

    def __init__(self, config: PretrainedConfig = None) -> None:
      super().__init__()
      self.config = config
      self.prefill_len = 0
      self.retained_key_cache: List[torch.Tensor] = []
      self.retained_value_cache: List[torch.Tensor] = []
      self.key_unit_cache: List[torch.Tensor] = []
      self.value_unit_cache: List[torch.Tensor] = []
      self.key_magnitude: List[torch.Tensor] = []
      self.value_magnitude: List[torch.Tensor] = []

      self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen
      self.mask_k = []
      self.mask_v = []

      self.hidden_states = []
      self.query_cache = []
      self.decode_q = []
      self.layer_map = []
      self.indices = []

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
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
        hidden_states: torch.Tensor = None, 
        query_states: torch.Tensor = None, 
        attention_mask = None,
        max_len = 256,
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
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.
            hidden_states (`torch.Tensor`, `optional`):
                The hidden states for the layer `layer_idx`.

        Return:
            A tuple containing the updated key, value states and hidden states.
        """
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        assert len(self.retained_key_cache) <= layer_idx
        self.retained_key_cache.append(key_states)
        self.retained_value_cache.append(value_states)
        self.hidden_states.append(None)

        sink_size = 8
        window_size = 8
        layer_map = []
        query_states = query_states[:,:,-window_size:,:]

        mask = torch.full((window_size, window_size), torch.finfo(key_states.dtype).min, device=key_states.device)
        mask_cond = torch.arange(mask.size(-1), device=key_states.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(key_states.device)
        attention_mask = mask[None, None, :, :]
        def f(a,n=256):
                return int(n/(10/32+22/32*(a+1)/2))
        ratio = 0.6
        scaled_size = min(f(ratio,max_len),self.retained_key_cache[0].shape[2])
        

        attn_lis = []
        prev_segment = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.retained_key_cache[0].shape[-1])
        prev_segment[:, :, -window_size:, -window_size:] += attention_mask
        attn_weights = prev_segment
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(self.retained_key_cache[0].dtype)
        attn_weights_sum_prev = attn_weights[:, :, -window_size:, sink_size:-window_size ].sum(dim = -2)
        self.query_cache.append(attn_weights_sum_prev)

        del query_states
        torch.cuda.empty_cache()
        

        self.key_unit_cache.append(None)
        self.value_unit_cache.append(None)
        self.key_magnitude.append(None)
        self.value_magnitude.append(None)
        self.mask_k.append(None)
        self.mask_v.append(None)

        if False:
            self.indices.append(None)
            # if layer_idx == 31:
            #     with open('layer_map_new.csv', 'r') as f:
            #             first_line = f.readline()
            #             num = int(first_line)
            #             for line in f:
            #                 isfirst = False
            #                 layer_map.append([i for i in line.strip().split(',')])
            #                 for i in range(5):
            #                     layer_map[-1][i] = int(layer_map[-1][i])
            #                 layer_map[-1][5] = float(layer_map[-1][5])
            #                 layer_map[-1][6] = float(layer_map[-1][6])
            layer_map = []
            if layer_idx == 31:
                for i in range(32):
                    print(i)
                    prev_segment = torch.matmul(self.query_cache[i], self.retained_key_cache[i].transpose(2, 3)) / math.sqrt(self.retained_key_cache[0].shape[-1])
                    p = prev_segment[:, :, -window_size:, :-window_size][0]  # [num_heads, seq_len, dim]
                    p_expanded = p.unsqueeze(1)  # [H_i, 1, S, D
                    for j in range(32):
                        if i >= j:
                            continue

                        # Get query-key pairs for both layers 
                        segment = torch.matmul(self.query_cache[j], self.retained_key_cache[j].transpose(2, 3)) / math.sqrt(self.retained_key_cache[0].shape[-1])
                        s = segment[:, :, -window_size:, :-window_size][0]  # [num_heads, seq_len, dim]
                        s_expanded = s.unsqueeze(0)  # [H_i, 1, S, D
 
                        cosine_sim = F.cosine_similarity(p_expanded, s_expanded, dim=-1)
                        cosine_sim_avg = cosine_sim.mean(dim=-1)  # [H_i, H_j]
                        # Find best matches for each head in layer i

                        for head_i in range(cosine_sim_avg.size(0)):
                            for head_j in range(cosine_sim_avg.size(1)):
                                sim = cosine_sim_avg[head_i][head_j].item()


                                # Calculate norm scaling for matched heads
                                p_head = p[head_i]
                                s_head = s[head_j]
                                p_norm = p_head.norm(dim=-1).mean().item()
                                s_norm = s_head.norm(dim=-1).mean().item()
                                scaling = s_norm / p_norm if p_norm != 0 else 0.0

                                # Store matched pair information
                                if sim < 0.9:
                                    continue
                                layer_map.append((i, j, 0, head_i, head_j, sim, scaling))

                        # Cleanup
                        del s,  s_expanded
                    del p, p_expanded
                layer_map.sort(key=lambda x:-x[-2])#from high to low
                used_segment = set()
                replaced_segment = set()
                print(len(layer_map))
                for item in layer_map:
                    i, j, seg,hi,hj, _, s = item
                    if len(replaced_segment)>= 22 * 32:
                        print(len(used_segment),len(replaced_segment))
                        break
                    if (j,seg,hj) in used_segment or (j,seg,hj) in replaced_segment or (i,seg,hi) in replaced_segment:
                        continue
                    # if j <= 2:
                    #     continue
                    # print('sim',i,j,hi,hj,_,s)
                    self.layer_map.append(item)
                    print(len(self.layer_map),item)
                    used_segment.add((i,seg,hi))
                    replaced_segment.add((j,seg,hj))
            if layer_idx == 31:
                with open('layer_map_final.csv', 'w') as f:
                    for item in self.layer_map:
                        f.write(','.join([str(i) for i in item]) + '\n')
                exit(0)
            return self.retained_key_cache[layer_idx], self.retained_value_cache[layer_idx], None
                # with open('layer_map.csv', 'r') as f:
                #     print('read')
                #     layer_map = []
                #     first_line = f.readline()
                #     num = int(first_line)
                #     for line in f:
                #         isfirst = False
                #         layer_map.append([i for i in line.strip().split(',')])
                #         for i in range(5):
                #             layer_map[-1][i] = int(layer_map[-1][i])
                #         layer_map[-1][5] = float(layer_map[-1][5])
                #         layer_map[-1][6] = float(layer_map[-1][6])
                
                # with open('layer_map.csv', 'w') as f:
                #     print('write')
                #     f.write(str(num+1) + '\n')
                #     if len(layer_map)!=0:
                #         for item, prev in zip(self.layer_map,layer_map):
                #             temp = [str(i) for i in item]
                #             temp[-1] = str((float(temp[-1]) + prev[-1]*num)/(num+1))
                #             temp[-2] = str((float(temp[-2]) + prev[-2]*num)/(num+1))
                #             f.write(','.join(temp) + '\n')
                #     else:
                #         for item in self.layer_map:
                #             f.write(','.join([str(i) for i in item]) + '\n')


            return ret_value[0], ret_value[1], ret_value[2]
        if layer_idx == 31:
            num_segments = 1
            segment_size = self.retained_key_cache[0].shape[2] // num_segments
            attn_diff = {}
            
            with open('layer_map_final.csv', 'r') as f:
                layer_map = []
                pair_map = {}
                for i in range(32):
                    pair_map[i] = []
                for line in f:
                    layer_map.append([i for i in line.strip().split(',')])
                    for i in range(5):
                        layer_map[-1][i] = int(layer_map[-1][i])
                    layer_map[-1][5] = float(layer_map[-1][5])
                    layer_map[-1][6] = float(layer_map[-1][6])
                    temp = layer_map[-1]
                    pair_map[temp[0]].append((temp[3], temp[1],temp[4]))

            # for i in range(32):
            #     prev_segment = torch.matmul(self.query_cache[i][:,:,-window_size:,:], self.retained_key_cache[i].transpose(2, 3)) / math.sqrt(self.retained_key_cache[0].shape[-1])
            #     prev_segment[:, :, -window_size:, -window_size:] += attention_mask
            #     attn_weights = prev_segment
            #     attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(self.retained_key_cache[i].dtype)
            #     attn_weights_sum_prev = attn_weights[:, :, -window_size:, sink_size:-window_size ].sum(dim = -2)
            #     attn_lis.append(
            #         attn_weights_sum_prev
            #     )
            attn_lis = self.query_cache
            for i in range(32):
                attn_diff[i] = None
                # min_num = (256-16)//2
                # max_num = 2*(256-16) - min_num

                # steps = (max_num - min_num) / 31
                # max_capacity_prompt = int(max_num - steps * i)
                # scaled_size = max_capacity_prompt + 16
                # scaled_size = 256
                # ratio = 0.6
                
                
                # def g(y,n=256):
                #     return (n/y-10/32)*2*32/22-1
                # prev_segment = torch.matmul(self.query_cache[i][:,:,-window_size:,:], self.retained_key_cache[i].transpose(2, 3)) / math.sqrt(self.retained_key_cache[0].shape[-1])
                # # p = prev_segment[:, :, -window_size:, :-window_size][0]  # [num_heads, seq_len, dim]
                # # p_expanded = p.unsqueeze(1)  # [H_i, 1, S, D]
                # prev_segment[:, :, -window_size:, -window_size:] += attention_mask
                # attn_weights = prev_segment
                # attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(self.retained_key_cache[i].dtype)
                # attn_weights_sum_prev = attn_weights[:, :, -window_size:, sink_size:-window_size ].sum(dim = -2)
                attn_weights_sum_prev = attn_lis[i]
                all_indices = (F.max_pool1d(attn_weights_sum_prev, kernel_size = 7, padding=7//2, stride=1)).topk((scaled_size-window_size-sink_size), dim=-1).indices #[1,h,10]


                h, n, d = attn_weights_sum_prev.shape  # Get dimensions
    
                # Expand dimensions for broadcasting
                # attn_weights_sum_prev: [h, n, d] -> [h, 1, n, d]
                # attn_weights_sum_prev_expanded = attn_weights_sum_prev.unsqueeze(1)
                
                diff = attn_weights_sum_prev.clone()
                counter = [1 for i in range(32)]
                for item in pair_map[i]:   
                    diff[:,item[0]:item[0]+1,:] += attn_lis[item[1]][:,item[2]:item[2]+1,:]
                    counter[item[0]] += 1
                # attn_diff[i] = diff-attn_weights_sum_prev
                attn_diff[i] = diff
                for index in range(32):
                    if counter != 0:
                        attn_diff[i][:,index:index+1,:]/=counter[index]
                # if counter != 0:
                #     attn_diff[i]/=counter
                # attn_diff[i] = attn_weights_sum_prev
                
                
                # # Compute pairwise absolute differences
                # # diffs: [h, h, n, d]
                # diffs = torch.abs(attn_weights_sum_prev_expanded - attn_weights_sum_prev.unsqueeze(0))
                
                # # Sum the differences along the head dimension (dim=1)
                # # result: [h, n, d]
                # attn_diff[i] = torch.sum(diffs, dim=1)
                

                # attn_diff[i] = F.max_pool1d(attn_diff[i], kernel_size = 7, padding=7//2, stride=1)
                selected_attn_diff =torch.gather(attn_diff[i], dim=-1, index=all_indices)
                indices = selected_attn_diff.topk((int(scaled_size*ratio) - window_size - sink_size), dim=-1).indices
                indices = torch.gather(all_indices, dim=-1, index=indices)

                final_indices_expanded = indices.unsqueeze(-1)  # Shape: [batch, heads, k_final, 1]
                all_indices_expanded = all_indices.unsqueeze(-2)      # Shape: [batch, heads, 1, k1]

                # Compare to find matches
                mask = (all_indices_expanded == final_indices_expanded).any(dim=-2)  # Shape: [batch, heads, k1]

                # 2. Filter out the indices in all_indices that are present in final_indices
                updated_all_indices = all_indices[~mask]  # Use the mask to exclude final_indices

                # Reshape updated_all_indices to maintain the original shape (excluding the removed indices)
                updated_all_indices = updated_all_indices.reshape(all_indices.shape[0], all_indices.shape[1], -1)
                
                self.indices.append((updated_all_indices + sink_size, indices + sink_size))
        # if layer_idx == 31:
        #     for i in attn_diff.keys():
        #         print(i,attn_diff[i].mean())
        if layer_idx == 31:
            temp_key = [i.clone() for i in self.retained_key_cache]
        # temp_value = [i.clone() for i in self.retained_value_cache]
        # Collect all indices and values for batched updates
        if layer_idx == 31:
            sink_indices = torch.arange(0, sink_size, device=self.retained_key_cache[0].device)
            window_indices = torch.arange(self.retained_key_cache[0].shape[-2] - window_size, self.retained_key_cache[0].shape[-2], device=self.retained_key_cache[0].device)
            combined_indices = torch.cat([sink_indices, window_indices])
            i_list, j_list, hi_list, hj_list, lis_list, _lis_list = [], [], [], [], [], []
            for item in layer_map:
                i, j, seg, hi, hj, _, s = item
                self.layer_map.append(item)
                i_list.append(i)
                j_list.append(j)
                hi_list.append(hi)
                hj_list.append(hj)
                lis_list.append(self.indices[j][1][0][hj])
                _lis_list.append(self.indices[i][0][0][hi])

            # Convert lists to tensors
            i_tensor = torch.tensor(i_list, device=self.retained_key_cache[0].device)
            j_tensor = torch.tensor(j_list, device=self.retained_key_cache[0].device)
            hi_tensor = torch.tensor(hi_list, device=self.retained_key_cache[0].device)
            hj_tensor = torch.tensor(hj_list, device=self.retained_key_cache[0].device)
            lis_tensor = lis_list
            _lis_tensor = _lis_list

            # Perform batched updates
            # Inside the batched updates loop where layer_map is processed
            for idx, (i, j, hi, hj, lis, _lis) in enumerate(zip(i_tensor, j_tensor, hi_tensor, hj_tensor, lis_tensor, _lis_tensor)):
                # Extract original heads from temporary clones
                p_head = temp_key[i][:, hi, _lis, :]  # [1, seq_len, dim]
                s_head = temp_key[j][:, hj, _lis, :]
                
                # Calculate norms and scaling factor
                p_norm = p_head.norm(dim=-1).mean().item()
                s_norm = s_head.norm(dim=-1).mean().item()
                scaling = s_norm / p_norm 
                
                # Apply scaling to the source key from layer i
                scaled_key = temp_key[i][:, hi, :, :] * scaling
                
                # Update retained key cache with scaled values
                self.retained_key_cache[j][:, hj, :, :] = scaled_key
                
                # Preserve specific indices from original head
                update_indices = torch.cat([combined_indices, lis])
                self.retained_key_cache[j][:, hj, update_indices, :] = temp_key[j][:, hj, update_indices, :]
                # self.retained_value_cache[j][:, hj, 128:-128, :] = temp_value[i][:, hi, 128:-128, :]
                # self.retained_key_cache[j][:, :, -8:, :] = temp_key[j][:, :, -8:, :]
                # self.retained_key_cache[j][:, :, :8, :] = temp_key[j][:, :, :8, :]
                # self.retained_value_cache[j][:, :, -8:, :] = temp_value[i][:, :, -8:, :]
        if layer_idx == 31:
            device = self.retained_key_cache[0].device
            
            # Process each layer individually
            for j in range(32):
                # Get sequence length for this layer
                seq_len = self.retained_key_cache[j].shape[-2]
                
                # Create combined indices for this layer
                window_start = seq_len - window_size
                window_indices = torch.arange(window_start, seq_len, device=device)
                combined_indices = torch.cat([sink_indices, window_indices])
                
                # Get layer-specific indices
                layer_indices_full = self.indices[j][1]  # [1, H, M]
                layer_indices_compress = self.indices[j][0]  # [1, H, M_compress]
                
                # Combine with sink/window indices
                combined_expanded = combined_indices.view(1, 1, -1).expand(1, layer_indices_full.size(1), -1)
                all_indices = torch.cat([layer_indices_full, combined_expanded], dim=-1)
                
                # Gather indices
                index_expanded = all_indices.unsqueeze(-1).expand(-1, -1, -1, self.retained_key_cache[j].size(-1))
                selected_keys = torch.gather(self.retained_key_cache[j], 2, index_expanded)
                selected_values = torch.gather(self.retained_value_cache[j], 2, index_expanded)
                
                # Gather compressed indices
                compress_index_expanded = layer_indices_compress.unsqueeze(-1).expand(-1, -1, -1, self.retained_key_cache[j].size(-1))
                unselected_keys = torch.gather(self.retained_key_cache[j], 2, compress_index_expanded)
                unselected_values = torch.gather(self.retained_value_cache[j], 2, compress_index_expanded)
                
                # Update caches for this layer
                self.key_unit_cache[j] = unselected_keys
                self.value_unit_cache[j] = unselected_values
                self.retained_key_cache[j] = selected_keys
                self.retained_value_cache[j] = selected_values
                self.indices[j] = None
            # item in self.layer_map is (i, j, seg, hi, hj, sim, scaling)
            # now we cast layer map to a dict with (j,hj) as key and (i,hi) as value.
            # Convert layer_map to a nested dictionary for per-layer storage
            layer_map = {}
            for _ in range(32):
                layer_map[_]={}
            for item in self.layer_map:
                i, j, seg, hi, hj, _, s = item
                if j not in layer_map:
                    layer_map[j] = {}  # Create a new dict for layer j
                layer_map[j][hj] = (i, hi)  # Store mapping per head

            self.layer_map = layer_map
            
            # delete all temporary variables only keep the final key and value caches
            del temp_key, self.query_cache
            torch.cuda.empty_cache()

        # layer_map.sort(key=lambda x:-x[-2])#from high to low

        ret_value = (self.retained_key_cache[layer_idx], self.retained_value_cache[layer_idx], None)
       
        return ret_value[0], ret_value[1], ret_value[2]
    def update_miniCache(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            unit_key_states: torch.Tensor,
            unit_value_states: torch.Tensor,
            key_magnitude: torch.Tensor,
            value_magnitude: torch.Tensor,
            mask_k,
            mask_v,
            previous_key_states: torch.Tensor,
            previous_value_states: torch.Tensor,
            layer_idx: int,   
            num_layers: int,   
    ):
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx` and the previous `key_states` and `value_states`.
        """

        if layer_idx < num_layers//4:
            self.key_unit_cache.append(None)
            self.value_unit_cache.append(None)
            self.key_magnitude.append(None)
            self.value_magnitude.append(None)
            self.mask_k.append(None)
            self.mask_v.append(None)
            return None
             
        if layer_idx % 2 == 1:
            # print('unit prefill:', layer_idx, unit_key_states)
            self.key_unit_cache.append(unit_key_states)
            self.value_unit_cache.append(unit_value_states)
            self.key_magnitude.append(key_magnitude)
            self.value_magnitude.append(value_magnitude)
            self.mask_k.append(mask_k)
            self.mask_v.append(mask_v)

            self.key_unit_cache.append(None)
            self.value_unit_cache.append(None)
            self.key_magnitude.append(None)
            self.value_magnitude.append(None)
            self.mask_k.append(None)
            self.mask_v.append(None)

            self.retained_key_cache[layer_idx] = key_states
            self.retained_value_cache[layer_idx] = value_states

            self.retained_key_cache[layer_idx-1] = previous_key_states
            self.retained_value_cache[layer_idx-1] = previous_value_states

    def update_miniCache_decode(self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        num_layers: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx` , also restore the kv cache for previous tokens.
        """
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        assert len(self.retained_key_cache) > layer_idx
        # print(self.retained_key_cache[layer_idx].shape, key_states.shape)

        # for item in self.layer_map:
        #         # print(item, len(past_key_value.decode_q)-1)
        #         if layer_idx == item[1]:
        #             # print(query_states.shape)
        #             key_states[:,item[3],:,:] = self.retained_key_cache[item[0]][:,item[3],-1,:]


       
        self.retained_key_cache[layer_idx] = torch.cat([self.retained_key_cache[layer_idx], key_states], dim=-2)
        self.retained_value_cache[layer_idx] = torch.cat([self.retained_value_cache[layer_idx], value_states], dim=-2)



        return self.retained_key_cache[layer_idx], self.retained_value_cache[layer_idx]
 
     
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
                cache.key_magnitude.append(past_key_values[layer_idx][4])
                cache.value_magnitude.append(past_key_values[layer_idx][5])
                cache.mask_k.append(past_key_values[layer_idx][6])
                cache.mask_v.append(past_key_values[layer_idx][7])
                cache.layer_map = past_key_values[layer_idx][8]
                cache.indices.append(past_key_values[layer_idx][9])


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
            legacy_cache += ((self.retained_key_cache[layer_idx],  self.retained_value_cache[layer_idx], self.key_unit_cache[layer_idx], self.value_unit_cache[layer_idx], self.key_magnitude[layer_idx], self.value_magnitude[layer_idx], self.mask_k[layer_idx], self.mask_v[layer_idx],self.layer_map, self.indices[layer_idx],),)
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
