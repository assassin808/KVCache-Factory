### Modified files
`cache_utils_minichace.py`, `llama_model.py`, `pyramidkv_utils.py`, and `monkeypatch.py`

---

**monkeypatch.py**: replace llama forward and attn_forward with minicache version

---

**llama_model.py**: add minicache forward and attn_forward function

---

**cache_utils_minicache.py**: this file rewrite the DynamicCache class.
there are 3 major function in updating KV cache:

*self.update*: It will take in and store key_states, value_states and hidden_states
*self.update_miniCache*: It will be used to generate minicache and will only be used in prefilling stage.
The cache (for compressed vector) will look like this:
```
self.key_unit_cache=[
  None,
  None,
  ...,
  compressed_k_16_17,
  None,
  compressed_k_18_19,
  None,
  ...,
]
```
The cache (for retained cache) will look like this:
```
self.retained_key_cache=[
  retained_0
  retained_1
  ...,
  retained_k_16
  retained_k_16,
  ...,
]
```
*self.update_miniCache*: It will be used to restore original key from minicache and will only be used in decoding stage

---

**pyramidkv_utils.py**

`MiniCacheKVCluster` class is defined in this function for compressing the KV cache to mini_cache

---

# Note
1. Currently, we store hidden states in th KV cache class, which are supposed to be discarded after use for efficiency.
2. Not all helping function in Dynamic class are rewritten, but is enough for current use.
3. Now, we only skip (8 = 32/4) layers before using minicache.