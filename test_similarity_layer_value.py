import os
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
import numpy as np
import csv

# /models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590/
# /models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659/
tokenizer = AutoTokenizer.from_pretrained(
    "/root/autodl-tmp/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659/"
)
model = AutoModelForCausalLM.from_pretrained(
    "/root/autodl-tmp/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659/",
)


# Input text
input_text = "Once upon a time, in a quaint little village nestled between rolling hills and lush forests, there existed a village named Green Valley, was home to villagers who lived peaceful and harmonious lives. A clear stream flowed through the village, nourishing the fields and orchards on its banks. Every morning, the villagers would rise early to begin their day’s work. The men worked in the fields, the women wove and cooked at home, and the children studied at the village school.In the village lived an elderly man named Grandpa Li, the village sage. Villagers would seek his advice whenever they faced difficulties. Grandpa Li was not only knowledgeable but also a great storyteller. Every evening, villagers would gather in front of his house to listen to his tales of ancient legends and magical adventures.One day, a stranger arrived in the village. He introduced himself as a traveler named Amin, carrying a mysterious package filled with strange items. The villagers, curious about Amin, gathered around to ask about his origins and experiences. Amin smiled and told them he came from a distant land in the East, having traveled to many places and witnessed many wonders.Amin decided to stay in the village, leading the villagers on explorations of the surrounding forests and hills, uncovering hidden treasures and secrets. Under his guidance, the villagers discovered rare flowers, plants, and animals they had never seen before. Amin also taught them new skills and knowledge, enriching their lives.However, Amin’s arrival was not by chance. He was guided by an ancient prophecy to Green Valley, seeking a legendary mystical power said to be hidden in the village. This power could only be found by those who were pure of heart and brave. Amin believed the villagers of Green Valley were the ones he was looking for.Amin decided to share this secret with the villagers. He gathered them under the big tree in the village and told them about the ancient prophecy. The villagers were both surprised and excited, eager to help Amin find this mystical power. Thus, under Amin’s leadership, they embarked on a journey filled with adventures and challenges.They traversed dense forests, climbed steep mountains, and overcame numerous obstacles. Eventually, they found the mystical power in a hidden cave. This power not only made the village more prosperous but also purified and elevated the villagers’ spirits.From then on, Green Valley became even more beautiful and harmonious, and the villagers grew more united and loving. Amin continued his journey, seeking more miracles and adventures. The story of Green Valley spread far and wide, told by Grandpa Li to all who would listen."

# Tokenize input
tokenized_input = tokenizer(input_text, return_tensors="pt", truncation=True)

# Forward pass with attention and past_key_values
with torch.no_grad():
    outputs = model(**tokenized_input, output_attentions=True, use_cache=True)

# Extract attentions and past_key_values
attentions = outputs.attentions  # List of attention tensors for each layer
past_key_values = outputs.past_key_values  # List of key-value states for each layer

# Get total number of layers
num_layers = model.config.num_hidden_layers

# Store attention outputs for the first head of each layer
layer_outputs = []

for layer_idx in range(num_layers):
    # Get attention weights and values for the first head
    attn_weights = attentions[layer_idx].squeeze(0)[1]  # (seq_len, seq_len)
    values = past_key_values[layer_idx][1].squeeze(0)[1]  # (seq_len, head_dim)
    print(attentions[layer_idx].shape,past_key_values[layer_idx][1].shape)
    
    # Compute attention output: (seq_len, head_dim)
    attn_output = attn_weights#torch.matmul(attn_weights, values)
    
    # Flatten and store
    layer_outputs.append(past_key_values[layer_idx][1])  # Flatten to (seq_len * head_dim,)

def cosine_similarity_matching_heads(v1, v2):
    """
    Compute the cosine similarity between v1 and v2 by matching the most similar heads.

    Args:
        v1 (torch.Tensor): Tensor of shape [1, 32, n, 128].
        v2 (torch.Tensor): Tensor of shape [1, 32, n, 128].

    Returns:
        float: Average cosine similarity after matching the most similar heads.
    """
    # Ensure the input tensors have the correct shape
    assert v1.shape == v2.shape, "Input tensors must have the same shape"
    assert v1.shape[0] == 1, "Batch size must be 1"
    # assert v1.shape[1] == 32, "Number of heads must be 32"

    # Reshape the tensors to [32, n, 128] by removing the batch dimension
    v1 = v1.squeeze(0)  # Shape: [32, n, 128]
    v2 = v2.squeeze(0)  # Shape: [32, n, 128]

    # Compute the cosine similarity between all pairs of heads
    # Reshape v1 to [32, 1, n, 128] and v2 to [1, 32, n, 128] for broadcasting
    v1_expanded = v1.unsqueeze(1)  # Shape: [32, 1, n, 128]
    v2_expanded = v2.unsqueeze(0)  # Shape: [1, 32, n, 128]

    # Compute cosine similarity along the last dimension (n, 128)
    cosine_sim = F.cosine_similarity(v1_expanded, v2_expanded, dim=-1)  # Shape: [32, 32, n]

    # Average over the sequence length (n) to get head-wise similarity
    cosine_sim = cosine_sim.mean(dim=-1)  # Shape: [32, 32]

    # Find the best matching head for each head in v1
    max_sim, _ = torch.max(cosine_sim, dim=1)  # Shape: [32]

    # Average the similarity scores of the best matching heads
    avg_similarity = max_sim.mean().item()

    return avg_similarity

# Calculate pairwise similarities
similarity_matrix = torch.zeros((num_layers, num_layers))
for i in range(num_layers):
    for j in range(num_layers):
        similarity_matrix[i, j] = cosine_similarity_matching_heads( layer_outputs[i], layer_outputs[j])

# Plot similarity matrix
plt.figure(figsize=(12, 10))
plt.imshow(similarity_matrix, cmap="viridis", vmin=-1, vmax=1)
plt.colorbar(label="Cosine Similarity")
plt.xlabel("Layer Index")
plt.ylabel("Layer Index")
plt.title("Attention Output Similarity (First Head) Across Layers")

# Add layer numbers as ticks
plt.xticks(range(num_layers))
plt.yticks(range(num_layers))

plt.grid(False)
plt.savefig("layer_similarity_matrix_first_head.png")
# plt.show()
print("Similarity matrix plot saved to layer_similarity_matrix_first_head.png")