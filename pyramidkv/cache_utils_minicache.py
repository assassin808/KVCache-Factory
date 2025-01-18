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
        self.hidden_states.append(hidden_states)

        layer_map = []

        if layer_idx == 31:
            num_segments = 5
            segment_size = self.retained_key_cache[0].shape[2] // num_segments
            
            for i in range(32):
                for j in range(32):
                    if i >= j:
                        continue
                    for seg in range(num_segments):  # Only compare segments at the same position
                        k_prev_segment = self.retained_key_cache[i][:, :, seg*segment_size:(seg+1)*segment_size, :]
                        k_segment = self.retained_key_cache[j][:, :, seg*segment_size:(seg+1)*segment_size, :]
                        k_similarity = torch.einsum("bhsd,bhsd->bhs", k_prev_segment, k_segment).mean().item()
                        layer_map.append((i, j, seg, k_similarity))  # Store layer indices, segment index, and similarity
        layer_map.sort(key=lambda x:x[-1])

        self.key_unit_cache.append(None)
        self.value_unit_cache.append(None)
        self.key_magnitude.append(None)
        self.value_magnitude.append(None)
        self.mask_k.append(None)
        self.mask_v.append(None)


        ret_value = (self.retained_key_cache[layer_idx], self.retained_value_cache[layer_idx], self.hidden_states[layer_idx])

        temp_key = self.retained_key_cache.copy()
        temp_value = self.retained_value_cache.copy()
        used_segment = set()
        replaced_segment = set()
        for item in layer_map:
            i, j, seg, _ = item
            if (j,seg) in used_segment or (j,seg) in replaced_segment or (i,seg) in used_segment:
                continue
            used_segment.add((i,seg))
            replaced_segment.add((j,seg))
            self.retained_key_cache[j][:, :, seg*segment_size:(seg+1)*segment_size, :] = temp_key[i][:, :, seg*segment_size:(seg+1)*segment_size, :]
            self.retained_value_cache[j][:, :, seg*segment_size:(seg+1)*segment_size, :] = temp_value[i][:, :, seg*segment_size:(seg+1)*segment_size, :]

        del temp_key
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
            legacy_cache += ((self.retained_key_cache[layer_idx],  self.retained_value_cache[layer_idx], self.key_unit_cache[layer_idx], self.value_unit_cache[layer_idx], self.key_magnitude[layer_idx], self.value_magnitude[layer_idx], self.mask_k[layer_idx], self.mask_v[layer_idx]),)
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
